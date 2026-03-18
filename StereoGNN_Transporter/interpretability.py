"""
Interpretability Module for StereoGNN
======================================

Provides mechanistic insights into model predictions:
1. Integrated Gradients for atom-level attribution
2. Attention weight analysis
3. Substructure importance
4. Embedding visualization (UMAP/t-SNE)
5. Stereocentre attention analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from captum.attr import IntegratedGradients, LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from model import StereoGNN
from featurizer import MoleculeGraphFeaturizer
from config import CONFIG


class IntegratedGradientsExplainer:
    """
    Compute atom-level attributions using Integrated Gradients.

    This method provides reliable attributions by integrating gradients
    along the path from a baseline to the input.
    """

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

    def _forward_func(self, x: Tensor, data: Data, target: str) -> Tensor:
        """Forward function for Integrated Gradients."""
        # Replace node features with the interpolated ones
        data.x = x
        batch = Batch.from_data_list([data]).to(self.device)

        output = self.model(batch)
        # Return substrate probability
        return F.softmax(output[target], dim=-1)[:, 2]

    def explain(
        self,
        smiles: str,
        target: str = 'DAT',
        n_steps: int = 50,
    ) -> Dict:
        """
        Compute atom-level attributions for a molecule.

        Args:
            smiles: SMILES string
            target: Target transporter ('DAT', 'NET', 'SERT')
            n_steps: Number of integration steps

        Returns:
            Dict with attributions and visualization data
        """
        self.model.eval()

        # Featurize molecule
        data = self.featurizer.featurize(smiles)
        if data is None:
            return None

        data = data.to(self.device)
        x = data.x.clone().requires_grad_(True)

        # Baseline (zeros)
        baseline = torch.zeros_like(x)

        # Integrated gradients
        attributions = torch.zeros_like(x)

        for step in range(n_steps):
            # Interpolate
            alpha = step / n_steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            data.x = interpolated
            batch = Batch.from_data_list([data]).to(self.device)

            output = self.model(batch)
            prob = F.softmax(output[target], dim=-1)[0, 2]

            # Backward pass
            self.model.zero_grad()
            prob.backward(retain_graph=True)

            # Accumulate gradients
            if interpolated.grad is not None:
                attributions += interpolated.grad

        # Scale by input - baseline
        attributions = attributions * (x - baseline) / n_steps

        # Aggregate to per-atom (sum over features)
        atom_attributions = attributions.sum(dim=-1).detach().cpu().numpy()

        # Normalize for visualization
        max_abs = np.abs(atom_attributions).max()
        if max_abs > 0:
            atom_attributions_norm = atom_attributions / max_abs
        else:
            atom_attributions_norm = atom_attributions

        return {
            'smiles': smiles,
            'target': target,
            'atom_attributions': atom_attributions,
            'atom_attributions_normalized': atom_attributions_norm,
            'num_atoms': len(atom_attributions),
        }


class AttentionAnalyzer:
    """
    Analyzes attention weights from the model.

    Identifies which atoms the model focuses on, particularly
    around stereocenters.
    """

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

    def get_attention_weights(
        self,
        smiles: str,
    ) -> Dict:
        """
        Extract attention weights for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Dict with attention weights and stereocenter info
        """
        self.model.eval()

        data = self.featurizer.featurize(smiles)
        if data is None:
            return None

        batch = Batch.from_data_list([data]).to(self.device)

        with torch.no_grad():
            output = self.model(batch, return_attention=True)

        # Get node attention from readout
        node_attention = output.get('node_attention')
        if node_attention is not None:
            node_attention = node_attention.cpu().numpy()

        # Identify stereocenters
        mol = Chem.MolFromSmiles(smiles)
        stereocenters = []
        if mol:
            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            stereocenters = [idx for idx, _ in chiral_centers]

        # Compute attention on stereocenters vs non-stereocenters
        if node_attention is not None and len(stereocenters) > 0:
            stereo_attention = node_attention[stereocenters].mean()
            non_stereo_mask = [i for i in range(len(node_attention)) if i not in stereocenters]
            if non_stereo_mask:
                non_stereo_attention = node_attention[non_stereo_mask].mean()
            else:
                non_stereo_attention = 0.0
            attention_ratio = stereo_attention / (non_stereo_attention + 1e-8)
        else:
            stereo_attention = 0.0
            non_stereo_attention = 0.0
            attention_ratio = 1.0

        return {
            'smiles': smiles,
            'node_attention': node_attention,
            'stereocenters': stereocenters,
            'stereo_attention_mean': stereo_attention,
            'non_stereo_attention_mean': non_stereo_attention,
            'attention_ratio': attention_ratio,  # >1 means more attention on stereocenters
        }


class SubstructureAnalyzer:
    """
    Identifies important substructures for prediction.

    Uses SMARTS patterns to check for known pharmacophores.
    """

    # Known pharmacophoric features for monoamine substrates
    PHARMACOPHORES = {
        'primary_amine': '[NH2;X3]',
        'secondary_amine': '[NH1;X3]',
        'tertiary_amine': '[NH0;X3]',
        'phenethylamine_core': 'c1ccccc1CCN',
        'alpha_methyl': 'CC(N)C',  # Alpha-methyl substitution
        'catechol': 'c1cc(O)c(O)cc1',
        'methylenedioxy': 'c1cc2OCOc2cc1',  # MDMA-like
        'tryptamine_core': 'c1[nH]c2ccccc2c1CCN',
        'tropane_core': 'C1CC2CCC1N2',  # Cocaine-like
    }

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Compile SMARTS patterns
        self.patterns = {}
        for name, smarts in self.PHARMACOPHORES.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                self.patterns[name] = pat

    def analyze(self, smiles: str, target: str = 'DAT') -> Dict:
        """
        Analyze substructure contributions.

        Args:
            smiles: SMILES string
            target: Target transporter

        Returns:
            Dict with substructure analysis
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Check which pharmacophores are present
        present_pharmacophores = {}
        for name, pattern in self.patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                present_pharmacophores[name] = {
                    'present': True,
                    'matches': matches,
                    'atom_indices': list(set(idx for match in matches for idx in match)),
                }

        # Get prediction
        data = self.featurizer.featurize(smiles)
        if data is None:
            return None

        batch = Batch.from_data_list([data]).to(self.device)

        with torch.no_grad():
            output = self.model(batch)
            probs = F.softmax(output[target], dim=-1)[0].cpu().numpy()

        return {
            'smiles': smiles,
            'target': target,
            'pharmacophores': present_pharmacophores,
            'prediction': {
                'inactive': probs[0],
                'blocker': probs[1],
                'substrate': probs[2],
            },
        }


class EmbeddingVisualizer:
    """
    Visualizes learned embeddings using dimensionality reduction.
    """

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

    def get_embeddings(self, smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings for a list of molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of (embeddings array, valid SMILES list)
        """
        self.model.eval()
        embeddings = []
        valid_smiles = []

        for smiles in smiles_list:
            data = self.featurizer.featurize(smiles)
            if data is None:
                continue

            batch = Batch.from_data_list([data]).to(self.device)

            with torch.no_grad():
                emb = self.model.get_embedding(batch)
                embeddings.append(emb.cpu().numpy())
                valid_smiles.append(smiles)

        if not embeddings:
            return np.array([]), []

        return np.vstack(embeddings), valid_smiles

    def visualize_umap(
        self,
        smiles_list: List[str],
        labels: Optional[List[int]] = None,
        label_names: Optional[Dict[int, str]] = None,
        title: str = "Molecular Embeddings",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize embeddings using UMAP.

        Args:
            smiles_list: List of SMILES strings
            labels: Optional labels for coloring
            label_names: Mapping from label to name
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required for UMAP visualization")

        embeddings, valid_smiles = self.get_embeddings(smiles_list)

        if len(embeddings) < 2:
            print("Not enough valid molecules for visualization")
            return None

        # UMAP reduction
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        embedding_2d = reducer.fit_transform(embeddings)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            # Filter labels to match valid molecules
            # Assuming labels align with original smiles_list
            valid_labels = []
            for i, smi in enumerate(smiles_list):
                if smi in valid_smiles:
                    valid_labels.append(labels[i])

            unique_labels = list(set(valid_labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = np.array(valid_labels) == label
                name = label_names.get(label, str(label)) if label_names else str(label)
                ax.scatter(
                    embedding_2d[mask, 0],
                    embedding_2d[mask, 1],
                    c=[color],
                    label=name,
                    alpha=0.7,
                    s=50,
                )
            ax.legend()
        else:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=50)

        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


class MoleculeHighlighter:
    """
    Creates molecule visualizations with atom highlighting.
    """

    @staticmethod
    def highlight_atoms(
        smiles: str,
        atom_weights: np.ndarray,
        title: str = "",
        cmap: str = 'RdBu_r',
    ) -> Optional[str]:
        """
        Create SVG of molecule with atoms highlighted by weights.

        Args:
            smiles: SMILES string
            atom_weights: Array of weights per atom
            title: Title for the image
            cmap: Colormap name

        Returns:
            SVG string or None if failed
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)

        # Normalize weights to [-1, 1]
        max_abs = np.abs(atom_weights).max()
        if max_abs > 0:
            weights_norm = atom_weights / max_abs
        else:
            weights_norm = atom_weights

        # Map to colors
        cmap_func = plt.cm.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=-1, vmax=1)

        atom_colors = {}
        atom_radii = {}
        for i, w in enumerate(weights_norm):
            rgba = cmap_func(norm(w))
            atom_colors[i] = rgba[:3]
            atom_radii[i] = 0.3 + 0.2 * abs(w)  # Vary radius by importance

        # Draw
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(range(mol.GetNumAtoms())),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
        )
        drawer.FinishDrawing()

        return drawer.GetDrawingText()


class InterpretabilityReport:
    """
    Generates comprehensive interpretability report for a prediction.
    """

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device

        self.ig_explainer = IntegratedGradientsExplainer(model, device)
        self.attention_analyzer = AttentionAnalyzer(model, device)
        self.substructure_analyzer = SubstructureAnalyzer(model, device)
        self.highlighter = MoleculeHighlighter()

    def generate_report(
        self,
        smiles: str,
        target: str = 'DAT',
    ) -> Dict:
        """
        Generate complete interpretability report.

        Args:
            smiles: SMILES string
            target: Target transporter

        Returns:
            Dict with all interpretability results
        """
        report = {
            'smiles': smiles,
            'target': target,
        }

        # Integrated gradients
        ig_result = self.ig_explainer.explain(smiles, target)
        if ig_result:
            report['integrated_gradients'] = ig_result

        # Attention analysis
        attn_result = self.attention_analyzer.get_attention_weights(smiles)
        if attn_result:
            report['attention'] = attn_result

        # Substructure analysis
        substruct_result = self.substructure_analyzer.analyze(smiles, target)
        if substruct_result:
            report['substructures'] = substruct_result

        # Generate visualization
        if ig_result:
            svg = self.highlighter.highlight_atoms(
                smiles,
                ig_result['atom_attributions_normalized'],
                title=f"{target} substrate attribution",
            )
            if svg:
                report['visualization_svg'] = svg

        return report


if __name__ == "__main__":
    print("=" * 60)
    print("Interpretability Module Test")
    print("=" * 60)

    # Create model and device
    model = StereoGNN()
    device = torch.device('cpu')

    # Test attention analyzer
    print("\nTesting Attention Analyzer...")
    attn = AttentionAnalyzer(model, device)
    result = attn.get_attention_weights("C[C@H](N)Cc1ccccc1")  # d-Amphetamine
    if result:
        print(f"  Stereocenters: {result['stereocenters']}")
        print(f"  Attention ratio (stereo/non-stereo): {result['attention_ratio']:.3f}")

    # Test substructure analyzer
    print("\nTesting Substructure Analyzer...")
    substruct = SubstructureAnalyzer(model, device)
    result = substruct.analyze("C[C@H](N)Cc1ccccc1", "DAT")
    if result:
        print(f"  Found pharmacophores: {list(result['pharmacophores'].keys())}")
        print(f"  Prediction: {result['prediction']}")

    # Test interpretability report
    print("\nGenerating interpretability report...")
    reporter = InterpretabilityReport(model, device)
    report = reporter.generate_report("C[C@H](N)Cc1ccccc1", "DAT")

    print(f"  Report sections: {list(report.keys())}")
    if 'visualization_svg' in report:
        print("  Visualization SVG generated successfully")
