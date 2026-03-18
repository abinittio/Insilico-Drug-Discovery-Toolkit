"""
StereoGNN Transporter Substrate Predictor - Ordinal Model (Continuous Scores)
==============================================================================

Predicts monoamine transporter activity (DAT, NET, SERT) for drug molecules.
Outputs continuous 0-1 scores for transport activity.
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import urllib.request
import urllib.parse
import json
import os
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem


def name_to_smiles(name: str) -> Optional[str]:
    """Convert molecule name to SMILES using PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/IsomericSMILES/JSON"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except:
        return None
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
import io
import base64
from PIL import Image

# Import the training featurizer
from featurizer import MoleculeGraphFeaturizer


# =============================================================================
# MODEL ARCHITECTURE - ORDINAL (Continuous 0-1 output)
# =============================================================================

class StereoGNNOrdinal(nn.Module):
    """Monoamine transporter substrate predictor with ordinal (continuous 0-1) output."""

    MONOAMINE_TARGETS = ['DAT', 'NET', 'SERT']

    def __init__(self, node_dim: int = 86, edge_dim: int = 18, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = GATv2Conv(
                self.hidden_dim, self.hidden_dim // self.num_heads,
                heads=self.num_heads, dropout=self.dropout,
                edge_dim=64, concat=True,
            )
            self.gnn_layers.append(conv)

        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

        # Task heads - ORDINAL: single sigmoid output (0-1 continuous score)
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(self.hidden_dim, 96),
                nn.LayerNorm(96),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(48, 1),  # Single output
                nn.Sigmoid(),  # Constrain to 0-1
            )
            for target in self.MONOAMINE_TARGETS
        })

    def forward(self, data, return_attention: bool = False):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for i, conv in enumerate(self.gnn_layers):
            x_new = conv(x, data.edge_index, edge_attr)
            x_new = self.norms[i](x_new)
            x = F.relu(x_new) + x

        graph_emb = global_mean_pool(x, data.batch)
        graph_emb = self.readout(graph_emb)

        output = {}
        for target in self.MONOAMINE_TARGETS:
            output[target] = self.heads[target](graph_emb)

        return output


# =============================================================================
# FEATURIZER (Self-contained)
# =============================================================================

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'num_hs': [0, 1, 2, 3, 4],
    'chiral_tag': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOANY,
    ],
}


def one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding


def get_atom_features(atom):
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    features += one_hot(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag'])

    # R/S configuration
    try:
        chiral_centers = Chem.FindMolChiralCenters(atom.GetOwningMol(), includeUnassigned=True)
        atom_idx = atom.GetIdx()
        rs_config = 0
        for idx, config in chiral_centers:
            if idx == atom_idx:
                rs_config = 1 if config == 'R' else (-1 if config == 'S' else 0)
                break
        features.append(rs_config)
    except:
        features.append(0)

    return features


def get_bond_features(bond):
    features = []
    features += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    features += one_hot(bond.GetStereo(), BOND_FEATURES['stereo'])
    features.append(1 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0)
    return features


def mol_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)

        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 18), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# PREDICTOR CLASS - ORDINAL
# =============================================================================

class TransporterPredictorOrdinal:
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = StereoGNNOrdinal(node_dim=86, edge_dim=18)

        # Use the same featurizer as training
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Load ordinal model weights
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Try different paths for model file
            model_paths = [
                os.path.join(script_dir, 'best_model_ordinal.pt'),  # Same directory (HuggingFace)
                'best_model_ordinal.pt',  # Current directory
                'outputs/best_model_ordinal.pt',  # Local development
            ]

            loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    self.loaded = True
                    loaded = True
                    print(f"Loaded ordinal model from {model_path}")
                    break

            if not loaded:
                print(f"Warning: Could not find model file in any of: {model_paths}")
                self.loaded = False
        except Exception as e:
            print(f"Warning: Could not load ordinal model: {e}")
            self.loaded = False

        self.targets = ['DAT', 'NET', 'SERT']

    def predict(self, smiles: str) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}

        # Use the training featurizer
        graph = self.featurizer.featurize(smiles)
        if graph is None:
            return {'error': 'Could not featurize molecule'}

        batch = Batch.from_data_list([graph])

        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)

        results = {}
        for target in self.targets:
            score = float(output[target][0, 0].numpy())
            # Classify based on score thresholds
            if score >= 0.66:
                prediction = 'substrate'
            elif score >= 0.33:
                prediction = 'blocker'
            else:
                prediction = 'inactive'

            results[target] = {
                'score': score,
                'prediction': prediction,
            }

        return results


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Global predictor instance
predictor = TransporterPredictorOrdinal()


def get_molecule_image(smiles: str) -> Optional[Image.Image]:
    """Generate molecule image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Draw molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_data))

    return img


def get_molecular_properties(smiles: str) -> Dict:
    """Calculate molecular properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    props = {
        'Molecular Weight': f"{Descriptors.MolWt(mol):.1f} Da",
        'LogP': f"{Descriptors.MolLogP(mol):.2f}",
        'TPSA': f"{Descriptors.TPSA(mol):.1f} A^2",
        'H-Bond Donors': Descriptors.NumHDonors(mol),
        'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
        'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
        'Aromatic Rings': Descriptors.NumAromaticRings(mol),
    }

    # Check for basic nitrogen
    has_basic_n = False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            if atom.GetTotalNumHs() > 0 or (atom.GetDegree() == 3 and not atom.GetIsAromatic()):
                has_basic_n = True
                break
    props['Basic Nitrogen'] = 'Yes' if has_basic_n else 'No'

    # Stereocenters
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    props['Stereocenters'] = len(chiral_centers)

    # CNS drug-likeness
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    cns_like = (100 <= mw <= 500 and -1 <= logp <= 5 and tpsa <= 90 and hbd <= 3 and hba <= 7)
    props['CNS Drug-like'] = 'Yes' if cns_like else 'No'

    return props


def get_score_bar(score: float, target: str) -> str:
    """Generate HTML for a score bar."""
    percentage = score * 100

    # Color gradient: gray (inactive) -> orange (blocker) -> green (substrate)
    if score < 0.33:
        color = '#9E9E9E'  # Gray
        category = 'Inactive'
    elif score < 0.66:
        color = '#FF9800'  # Orange
        category = 'Blocker'
    else:
        color = '#4CAF50'  # Green
        category = 'Substrate'

    bar_html = f'''
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-weight: bold;">{target}</span>
            <span style="color: {color}; font-weight: bold;">{score:.3f} ({category})</span>
        </div>
        <div style="background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
            <div style="background: linear-gradient(to right, #9E9E9E 0%, #9E9E9E 33%, #FF9800 33%, #FF9800 66%, #4CAF50 66%, #4CAF50 100%);
                        height: 100%; position: relative;">
                <div style="position: absolute; left: {percentage}%; top: 0; bottom: 0; width: 3px; background: black;"></div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #666; margin-top: 2px;">
            <span>0 (Inactive)</span>
            <span>0.33</span>
            <span>0.66</span>
            <span>1 (Substrate)</span>
        </div>
    </div>
    '''
    return bar_html


def predict_transporter(input_text: str):
    """Main prediction function for Gradio. Accepts SMILES or molecule name."""
    if not input_text or not input_text.strip():
        return None, "Please enter a SMILES string or molecule name", "", ""

    input_text = input_text.strip()

    # Try as SMILES first
    mol = Chem.MolFromSmiles(input_text)
    smiles = input_text
    name_used = None

    if mol is None:
        # Try as molecule name via PubChem
        smiles = name_to_smiles(input_text)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            name_used = input_text

    if mol is None:
        return None, f"Could not parse '{input_text}' as SMILES or find it in PubChem", "", ""

    # Get molecule image
    mol_img = get_molecule_image(smiles)

    # Get predictions
    results = predictor.predict(smiles)

    if 'error' in results:
        return mol_img, f"Error: {results['error']}", "", ""

    # Format prediction results with score bars
    pred_html = "<div style='font-family: Arial, sans-serif;'>"
    if name_used:
        pred_html += f"<p style='color: #2196F3;'><b>{name_used}</b> resolved to: <code>{smiles}</code></p>"
    pred_html += "<h3>Transport Activity Scores</h3>"
    pred_html += "<p style='font-size: 12px; color: #666;'>Continuous scores from 0 (inactive) to 1 (substrate)</p>"

    substrate_targets = []
    blocker_targets = []

    for target in ['DAT', 'NET', 'SERT']:
        r = results[target]
        score = r['score']
        pred = r['prediction']

        pred_html += get_score_bar(score, target)

        if pred == 'substrate':
            substrate_targets.append(target)
        elif pred == 'blocker':
            blocker_targets.append(target)

    # Summary table
    pred_html += "<h4 style='margin-top: 20px;'>Summary</h4>"
    pred_html += "<table style='width:100%; border-collapse: collapse; font-size: 14px;'>"
    pred_html += "<tr style='background:#f5f5f5;'><th style='padding:8px; text-align:left;'>Target</th><th>Score</th><th>Category</th></tr>"

    for target in ['DAT', 'NET', 'SERT']:
        r = results[target]
        score = r['score']
        pred = r['prediction']
        colors = {'substrate': '#4CAF50', 'blocker': '#FF9800', 'inactive': '#9E9E9E'}
        icons = {'substrate': '🟢', 'blocker': '🟡', 'inactive': '⚪'}

        pred_html += f"<tr>"
        pred_html += f"<td style='padding:8px; font-weight:bold;'>{target}</td>"
        pred_html += f"<td style='padding:8px; text-align:center;'>{score:.3f}</td>"
        pred_html += f"<td style='padding:8px; color:{colors[pred]}; font-weight:bold;'>{icons[pred]} {pred.upper()}</td>"
        pred_html += f"</tr>"

    pred_html += "</table>"

    # Category
    if len(substrate_targets) == 0:
        if len(blocker_targets) > 0:
            category = f"BLOCKER ({', '.join(blocker_targets)})"
            cat_color = "#FF9800"
        else:
            category = "INACTIVE"
            cat_color = "#9E9E9E"
    elif len(substrate_targets) == 3:
        category = "PAN-SUBSTRATE (DAT, NET, SERT)"
        cat_color = "#4CAF50"
    else:
        category = f"SELECTIVE SUBSTRATE ({', '.join(substrate_targets)})"
        cat_color = "#2196F3"

    pred_html += f"<p style='margin-top:16px;'><strong>Category:</strong> <span style='color:{cat_color}; font-weight:bold;'>{category}</span></p>"
    pred_html += "</div>"

    # Molecular properties
    props = get_molecular_properties(smiles)
    props_html = "<div style='font-family: Arial, sans-serif;'>"
    props_html += "<h3>Molecular Properties</h3>"
    props_html += "<table style='width:100%; border-collapse: collapse;'>"

    for key, val in props.items():
        props_html += f"<tr><td style='padding:4px;'>{key}</td><td style='padding:4px; font-weight:bold;'>{val}</td></tr>"

    props_html += "</table></div>"

    # Interpretation
    interp = "<div style='font-family: Arial, sans-serif;'>"
    interp += "<h3>Interpretation</h3>"

    # Get average score
    avg_score = np.mean([results[t]['score'] for t in ['DAT', 'NET', 'SERT']])
    interp += f"<p><strong>Average Transport Score:</strong> {avg_score:.3f}</p>"

    if len(substrate_targets) > 0:
        interp += f"<p>This molecule is predicted to be a <strong>substrate</strong> for {', '.join(substrate_targets)}.</p>"
        interp += "<p>Substrates are actively transported across the cell membrane by the transporter protein.</p>"
    elif len(blocker_targets) > 0:
        interp += f"<p>This molecule may act as a <strong>blocker</strong> at {', '.join(blocker_targets)}.</p>"
        interp += "<p>Blockers inhibit transporter function without being transported themselves.</p>"
    else:
        interp += "<p>This molecule is <strong>not predicted to interact significantly</strong> with monoamine transporters.</p>"

    # Score interpretation guide
    interp += """
    <div style='margin-top: 16px; padding: 12px; background: #f5f5f5; border-radius: 8px;'>
        <strong>Score Interpretation:</strong>
        <ul style='margin: 8px 0; padding-left: 20px;'>
            <li><strong>0.00 - 0.33:</strong> Inactive (no significant interaction)</li>
            <li><strong>0.33 - 0.66:</strong> Blocker (inhibits without being transported)</li>
            <li><strong>0.66 - 1.00:</strong> Substrate (actively transported)</li>
        </ul>
    </div>
    """

    # Warnings
    warnings = []
    if props.get('Basic Nitrogen') == 'No':
        warnings.append("No basic nitrogen detected (unusual for substrates)")
    if props.get('CNS Drug-like') == 'No':
        warnings.append("Does not meet CNS drug-likeness criteria")
    if props.get('Stereocenters', 0) > 0:
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for idx, config in chiral_centers:
            if config == '?':
                warnings.append("Contains undefined stereocenters - predictions may vary by enantiomer")
                break

    if warnings:
        interp += "<p style='color:#FF5722; margin-top: 12px;'><strong>Warnings:</strong></p><ul>"
        for w in warnings:
            interp += f"<li>{w}</li>"
        interp += "</ul>"

    interp += "</div>"

    return mol_img, pred_html, props_html, interp


# Example molecules
EXAMPLES = [
    ["C[C@H](N)Cc1ccccc1", "d-Amphetamine (DAT/NET substrate)"],
    ["C[C@@H](N)Cc1ccccc1", "l-Amphetamine (less active)"],
    ["C[C@H](NC)Cc1ccccc1", "d-Methamphetamine"],
    ["NCCc1ccc(O)c(O)c1", "Dopamine (natural DAT substrate)"],
    ["NC[C@H](O)c1ccc(O)c(O)c1", "Norepinephrine (NET substrate)"],
    ["NCCc1c[nH]c2ccc(O)cc12", "Serotonin (SERT substrate)"],
    ["C[C@H](NC)Cc1ccc2OCOc2c1", "MDMA (pan-substrate)"],
    ["COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C", "Cocaine (blocker)"],
    ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Caffeine (inactive)"],
]


# Build interface
with gr.Blocks(title="StereoGNN Transporter Predictor (Ordinal)") as demo:
    gr.Markdown("""
    # StereoGNN Transporter Substrate Predictor (Ordinal Model)

    Predict monoamine transporter activity for drug molecules using a stereochemistry-aware graph neural network.

    **This model outputs continuous transport activity scores (0-1) instead of discrete classes.**

    **Targets:**
    - **DAT** - Dopamine Transporter
    - **NET** - Norepinephrine Transporter
    - **SERT** - Serotonin Transporter

    **Score Interpretation:**
    - **0.00 - 0.33:** Inactive (no significant interaction)
    - **0.33 - 0.66:** Blocker (inhibits without being transported)
    - **0.66 - 1.00:** Substrate (actively transported)

    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            smiles_input = gr.Textbox(
                label="Enter SMILES or Drug Name",
                placeholder="e.g., C[C@H](N)Cc1ccccc1 or 'amphetamine'",
                lines=1,
            )
            predict_btn = gr.Button("Predict", variant="primary")

            gr.Markdown("### Example Molecules")
            example_btns = []
            for smiles, name in EXAMPLES:
                btn = gr.Button(name, size="sm")
                btn.click(fn=lambda s=smiles: s, outputs=smiles_input)

        with gr.Column(scale=1):
            mol_image = gr.Image(label="Molecule Structure", type="pil")

    with gr.Row():
        with gr.Column():
            prediction_output = gr.HTML(label="Predictions")
        with gr.Column():
            properties_output = gr.HTML(label="Properties")

    interpretation_output = gr.HTML(label="Interpretation")

    predict_btn.click(
        fn=predict_transporter,
        inputs=smiles_input,
        outputs=[mol_image, prediction_output, properties_output, interpretation_output]
    )

    smiles_input.submit(
        fn=predict_transporter,
        inputs=smiles_input,
        outputs=[mol_image, prediction_output, properties_output, interpretation_output]
    )

    gr.Markdown("""
    ---

    ### About

    This model uses a Graph Neural Network (GNN) with explicit stereochemistry encoding to predict
    transport activity at monoamine transporters.

    **Model Performance (Ordinal):**
    - Mean Spearman Correlation: 0.893
    - DAT AUC: 0.990 | NET AUC: 0.992 | SERT AUC: 0.990
    - Stereo Sensitivity: 83.3% (correctly distinguishes enantiomers)

    **Stereo Examples:**
    - d-Amphetamine: 0.902 vs l-Amphetamine: 0.624 (correctly predicts d > l)

    **Citation:** If you use this tool, please cite our work.

    **Disclaimer:** This is a research tool. Predictions should be validated experimentally.
    """)


if __name__ == "__main__":
    demo.launch(share=False)
