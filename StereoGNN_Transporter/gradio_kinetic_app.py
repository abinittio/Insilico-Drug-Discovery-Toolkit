"""
Gradio Interface for StereoGNN Kinetic Transporter Predictor
=============================================================

Interactive web interface for predicting:
- Transporter substrate/blocker classification (DAT, NET, SERT)
- Kinetic parameters (pKi, pIC50)
- Interaction mode (substrate, competitive, non-competitive, partial)
- Uncertainty quantification

Usage:
    python gradio_kinetic_app.py
    # Opens at http://localhost:7860
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

import gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import KineticTransporterPredictor, KineticTransporterPrediction
from config import CONFIG


# Global predictor instance (loaded once)
PREDICTOR: Optional[KineticTransporterPredictor] = None


def load_predictor(model_path: Optional[str] = None) -> KineticTransporterPredictor:
    """Load or get the global predictor instance."""
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = KineticTransporterPredictor(model_path=model_path)
    return PREDICTOR


def validate_smiles(smiles: str) -> Tuple[bool, str, Optional[object]]:
    """Validate SMILES string and return molecule object."""
    if not smiles or not smiles.strip():
        return False, "Please enter a SMILES string", None

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False, f"Invalid SMILES: '{smiles}'", None

    return True, "Valid SMILES", mol


def draw_molecule(smiles: str) -> Optional[object]:
    """Draw molecule from SMILES."""
    valid, msg, mol = validate_smiles(smiles)
    if not valid:
        return None

    try:
        img = Draw.MolToImage(mol, size=(300, 300))
        return img
    except Exception:
        return None


def format_prediction_html(pred: KineticTransporterPrediction) -> str:
    """Format prediction results as HTML."""
    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;">
            Prediction Results
        </h3>
        <p><strong>SMILES:</strong> <code>{pred.smiles}</code></p>
    """

    for target in ['DAT', 'NET', 'SERT']:
        target_pred = getattr(pred, target)
        if target_pred is None:
            continue

        # Determine color based on classification
        class_idx = target_pred.predicted_class
        class_colors = {0: '#27ae60', 1: '#e74c3c', 2: '#95a5a6'}  # substrate, blocker, inactive
        class_names = {0: 'Substrate', 1: 'Blocker', 2: 'Inactive'}
        class_color = class_colors.get(class_idx, '#95a5a6')
        class_name = class_names.get(class_idx, 'Unknown')

        # Confidence bar
        confidence = target_pred.confidence * 100
        conf_color = '#27ae60' if confidence > 70 else '#f39c12' if confidence > 50 else '#e74c3c'

        # Mode name
        mode_name = target_pred.kinetics.interaction_mode if target_pred.kinetics else 'N/A'

        # pKi formatting
        pki = target_pred.kinetics.pKi_mean if target_pred.kinetics else 0
        pki_std = target_pred.kinetics.pKi_total_std if target_pred.kinetics else 0

        # pIC50 formatting
        pic50 = target_pred.kinetics.pIC50_mean if target_pred.kinetics else 0
        pic50_std = target_pred.kinetics.pIC50_total_std if target_pred.kinetics else 0

        # Kinetic bias
        bias = target_pred.kinetics.kinetic_bias_mean if target_pred.kinetics else 0.5
        bias_label = 'Uptake' if bias > 0.5 else 'Blockade'

        html += f"""
        <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {class_color};">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">
                {target} Transporter
                <span style="background: {class_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 10px;">
                    {class_name}
                </span>
            </h4>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div>
                    <strong>Confidence:</strong>
                    <div style="background: #ecf0f1; border-radius: 4px; height: 20px; margin-top: 3px;">
                        <div style="background: {conf_color}; width: {confidence:.0f}%; height: 100%; border-radius: 4px; text-align: center; color: white; font-size: 0.8em; line-height: 20px;">
                            {confidence:.1f}%
                        </div>
                    </div>
                </div>
                <div>
                    <strong>Interaction Mode:</strong> {mode_name}
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px;">
                <div style="text-align: center; padding: 8px; background: white; border-radius: 4px;">
                    <div style="font-size: 0.8em; color: #7f8c8d;">pKi</div>
                    <div style="font-size: 1.2em; font-weight: bold;">{pki:.2f}</div>
                    <div style="font-size: 0.7em; color: #95a5a6;">&plusmn; {pki_std:.2f}</div>
                </div>
                <div style="text-align: center; padding: 8px; background: white; border-radius: 4px;">
                    <div style="font-size: 0.8em; color: #7f8c8d;">pIC50</div>
                    <div style="font-size: 1.2em; font-weight: bold;">{pic50:.2f}</div>
                    <div style="font-size: 0.7em; color: #95a5a6;">&plusmn; {pic50_std:.2f}</div>
                </div>
                <div style="text-align: center; padding: 8px; background: white; border-radius: 4px;">
                    <div style="font-size: 0.8em; color: #7f8c8d;">Kinetic Bias</div>
                    <div style="font-size: 1.2em; font-weight: bold;">{bias:.2f}</div>
                    <div style="font-size: 0.7em; color: #95a5a6;">{bias_label}</div>
                </div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def format_comparison_html(comparison: Dict) -> str:
    """Format enantiomer comparison as HTML."""
    html = """
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <h3 style="color: #9b59b6; border-bottom: 2px solid #9b59b6; padding-bottom: 5px;">
            Stereoisomer Comparison
        </h3>
    """

    for target in ['DAT', 'NET', 'SERT']:
        if target not in comparison:
            continue

        data = comparison[target]
        pki_diff = data.get('pKi_difference', 0)
        mode_same = data.get('same_mode', True)

        # Determine significance color
        if abs(pki_diff) > 0.5:
            diff_color = '#e74c3c'  # Significant
            diff_label = 'Significant stereo-selectivity'
        elif abs(pki_diff) > 0.2:
            diff_color = '#f39c12'  # Moderate
            diff_label = 'Moderate stereo-selectivity'
        else:
            diff_color = '#27ae60'  # Similar
            diff_label = 'Similar activity'

        html += f"""
        <div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 8px;">
            <strong>{target}:</strong>
            <span style="margin-left: 10px; padding: 3px 8px; background: {diff_color}; color: white; border-radius: 4px;">
                &Delta;pKi = {pki_diff:+.2f}
            </span>
            <span style="margin-left: 10px; color: #7f8c8d;">({diff_label})</span>
            <span style="margin-left: 10px; color: {'#27ae60' if mode_same else '#e74c3c'};">
                {'Same' if mode_same else 'Different'} mode
            </span>
        </div>
        """

    html += "</div>"
    return html


def predict_single(smiles: str) -> Tuple[object, str]:
    """Predict for a single molecule."""
    valid, msg, mol = validate_smiles(smiles)
    if not valid:
        return None, f"<p style='color: red;'>{msg}</p>"

    try:
        predictor = load_predictor()
        pred = predictor.predict(smiles.strip())

        if pred is None:
            return None, "<p style='color: red;'>Prediction failed. Check SMILES validity.</p>"

        img = draw_molecule(smiles)
        html = format_prediction_html(pred)

        return img, html

    except Exception as e:
        return None, f"<p style='color: red;'>Error: {str(e)}</p>"


def compare_stereoisomers(smiles1: str, smiles2: str) -> Tuple[object, object, str, str]:
    """Compare two stereoisomers."""
    valid1, msg1, mol1 = validate_smiles(smiles1)
    valid2, msg2, mol2 = validate_smiles(smiles2)

    if not valid1:
        return None, None, f"<p style='color: red;'>Molecule 1: {msg1}</p>", ""
    if not valid2:
        return None, None, "", f"<p style='color: red;'>Molecule 2: {msg2}</p>"

    try:
        predictor = load_predictor()

        # Predict both
        pred1 = predictor.predict(smiles1.strip())
        pred2 = predictor.predict(smiles2.strip())

        if pred1 is None or pred2 is None:
            return None, None, "<p style='color: red;'>Prediction failed</p>", ""

        # Compare
        comparison = predictor.compare_enantiomers(smiles1.strip(), smiles2.strip())

        img1 = draw_molecule(smiles1)
        img2 = draw_molecule(smiles2)
        html1 = format_prediction_html(pred1)
        html2 = format_prediction_html(pred2) + format_comparison_html(comparison)

        return img1, img2, html1, html2

    except Exception as e:
        return None, None, f"<p style='color: red;'>Error: {str(e)}</p>", ""


def batch_predict(smiles_text: str) -> str:
    """Predict for multiple molecules (one per line)."""
    lines = [line.strip() for line in smiles_text.strip().split('\n') if line.strip()]

    if not lines:
        return "<p style='color: red;'>Please enter at least one SMILES string</p>"

    if len(lines) > 50:
        return "<p style='color: red;'>Maximum 50 molecules per batch</p>"

    try:
        predictor = load_predictor()
        predictions = predictor.predict_batch(lines)

        html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h3>Batch Results ({len(predictions)} molecules)</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #3498db; color: white;">
                    <th style="padding: 8px; text-align: left;">SMILES</th>
                    <th style="padding: 8px;">DAT</th>
                    <th style="padding: 8px;">NET</th>
                    <th style="padding: 8px;">SERT</th>
                    <th style="padding: 8px;">Best Target</th>
                </tr>
        """

        class_names = {0: 'Sub', 1: 'Block', 2: 'Inact'}
        class_colors = {0: '#27ae60', 1: '#e74c3c', 2: '#95a5a6'}

        for i, pred in enumerate(predictions):
            if pred is None:
                html += f"""
                <tr style="background: {'#f8f9fa' if i % 2 == 0 else 'white'};">
                    <td style="padding: 8px;" colspan="5">Invalid SMILES: {lines[i]}</td>
                </tr>
                """
                continue

            # Find best target
            best_target = 'N/A'
            best_pki = -999
            for target in ['DAT', 'NET', 'SERT']:
                tp = getattr(pred, target)
                if tp and tp.kinetics and tp.kinetics.pKi_mean > best_pki:
                    best_pki = tp.kinetics.pKi_mean
                    best_target = target

            row_bg = '#f8f9fa' if i % 2 == 0 else 'white'
            html += f'<tr style="background: {row_bg};">'
            html += f'<td style="padding: 8px; font-family: monospace; font-size: 0.85em;">{pred.smiles[:30]}...</td>'

            for target in ['DAT', 'NET', 'SERT']:
                tp = getattr(pred, target)
                if tp:
                    cls = tp.predicted_class
                    color = class_colors.get(cls, '#95a5a6')
                    name = class_names.get(cls, '?')
                    pki = tp.kinetics.pKi_mean if tp.kinetics else 0
                    html += f'<td style="padding: 8px; text-align: center;"><span style="background: {color}; color: white; padding: 2px 6px; border-radius: 3px;">{name}</span><br><small>pKi: {pki:.1f}</small></td>'
                else:
                    html += '<td style="padding: 8px; text-align: center;">-</td>'

            html += f'<td style="padding: 8px; text-align: center; font-weight: bold;">{best_target}</td>'
            html += '</tr>'

        html += '</table></div>'
        return html

    except Exception as e:
        return f"<p style='color: red;'>Error: {str(e)}</p>"


# Example molecules
EXAMPLE_MOLECULES = {
    "d-Amphetamine (DAT substrate)": "C[C@H](N)Cc1ccccc1",
    "l-Amphetamine (less active)": "C[C@@H](N)Cc1ccccc1",
    "d-Methamphetamine": "C[C@H](NC)Cc1ccccc1",
    "MDMA (S-isomer, SERT)": "C[C@H](NC)Cc1ccc2OCOc2c1",
    "Dopamine (endogenous)": "NCCc1ccc(O)c(O)c1",
    "Norepinephrine (endogenous)": "NC[C@H](O)c1ccc(O)c(O)c1",
    "Serotonin (endogenous)": "NCCc1c[nH]c2ccc(O)cc12",
    "Cocaine (blocker)": "COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3",
}


def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="StereoGNN Kinetic Predictor") as demo:

        gr.Markdown("""
        # StereoGNN Kinetic Transporter Predictor

        Predict monoamine transporter interactions with **stereochemistry-aware** deep learning.

        **Targets:** DAT (Dopamine), NET (Norepinephrine), SERT (Serotonin)

        **Outputs:** Classification, pKi, pIC50, Interaction Mode, Uncertainty
        """)

        with gr.Tabs():

            # Tab 1: Single molecule prediction
            with gr.TabItem("Single Molecule"):
                with gr.Row():
                    with gr.Column(scale=1):
                        smiles_input = gr.Textbox(
                            label="SMILES",
                            placeholder="Enter SMILES string...",
                            lines=2,
                        )

                        gr.Markdown("**Quick Examples:**")
                        example_dropdown = gr.Dropdown(
                            choices=list(EXAMPLE_MOLECULES.keys()),
                            label="Select example",
                            interactive=True,
                        )

                        predict_btn = gr.Button("Predict", variant="primary")

                    with gr.Column(scale=1):
                        mol_image = gr.Image(label="Structure", type="pil")

                result_html = gr.HTML(label="Results")

                # Event handlers
                example_dropdown.change(
                    lambda x: EXAMPLE_MOLECULES.get(x, ""),
                    inputs=[example_dropdown],
                    outputs=[smiles_input],
                )

                predict_btn.click(
                    predict_single,
                    inputs=[smiles_input],
                    outputs=[mol_image, result_html],
                )

                smiles_input.submit(
                    predict_single,
                    inputs=[smiles_input],
                    outputs=[mol_image, result_html],
                )

            # Tab 2: Stereoisomer comparison
            with gr.TabItem("Compare Stereoisomers"):
                gr.Markdown("""
                Compare two stereoisomers to analyze stereo-selectivity.

                *Tip: Use d/l or R/S isomers of the same compound*
                """)

                with gr.Row():
                    with gr.Column():
                        smiles1 = gr.Textbox(
                            label="Stereoisomer 1",
                            placeholder="e.g., C[C@H](N)Cc1ccccc1",
                            value="C[C@H](N)Cc1ccccc1",
                        )
                        img1 = gr.Image(label="Structure 1", type="pil")
                        result1 = gr.HTML()

                    with gr.Column():
                        smiles2 = gr.Textbox(
                            label="Stereoisomer 2",
                            placeholder="e.g., C[C@@H](N)Cc1ccccc1",
                            value="C[C@@H](N)Cc1ccccc1",
                        )
                        img2 = gr.Image(label="Structure 2", type="pil")
                        result2 = gr.HTML()

                compare_btn = gr.Button("Compare", variant="primary")

                compare_btn.click(
                    compare_stereoisomers,
                    inputs=[smiles1, smiles2],
                    outputs=[img1, img2, result1, result2],
                )

            # Tab 3: Batch prediction
            with gr.TabItem("Batch Prediction"):
                gr.Markdown("""
                Predict multiple molecules at once (one SMILES per line, max 50).
                """)

                batch_input = gr.Textbox(
                    label="SMILES List",
                    placeholder="Enter one SMILES per line...",
                    lines=10,
                    value="""C[C@H](N)Cc1ccccc1
C[C@@H](N)Cc1ccccc1
NCCc1ccc(O)c(O)c1
C[C@H](NC)Cc1ccccc1""",
                )

                batch_btn = gr.Button("Predict All", variant="primary")
                batch_result = gr.HTML()

                batch_btn.click(
                    batch_predict,
                    inputs=[batch_input],
                    outputs=[batch_result],
                )

            # Tab 4: About
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About StereoGNN Kinetic Predictor

                This tool uses a **stereochemistry-aware graph neural network** to predict
                interactions with monoamine transporters:

                - **DAT** (Dopamine Transporter)
                - **NET** (Norepinephrine Transporter)
                - **SERT** (Serotonin Transporter)

                ### Predictions Include:

                | Output | Description |
                |--------|-------------|
                | Classification | Substrate, Blocker, or Inactive |
                | pKi | Binding affinity (-log10 Ki) |
                | pIC50 | Functional potency (-log10 IC50) |
                | Interaction Mode | Substrate, Competitive, Non-competitive, Partial |
                | Kinetic Bias | Uptake vs blockade preference |
                | Uncertainty | Confidence intervals from MC Dropout |

                ### Key Features:

                - **Stereochemistry encoding**: R/S configurations affect transporter selectivity
                - **Multi-task learning**: Shared representation for all three transporters
                - **Uncertainty quantification**: Know when the model is uncertain
                - **Interpretable**: Attention weights highlight important atoms

                ### Reference Compounds:

                | Compound | Type | Primary Target |
                |----------|------|----------------|
                | d-Amphetamine | Substrate | DAT > NET >> SERT |
                | d-Methamphetamine | Substrate | DAT > NET |
                | MDMA | Substrate | SERT > DAT |
                | Cocaine | Blocker | DAT = NET = SERT |
                | Methylphenidate | Blocker | DAT > NET |

                ### Technical Details:

                - Architecture: GAT-based GNN with stereo-aware encoders
                - Training: Multi-task focal loss with heteroscedastic uncertainty
                - Uncertainty: MC Dropout (epistemic) + learned variance (aleatoric)
                """)

        gr.Markdown("""
        ---
        *StereoGNN Kinetic Predictor - For research use only*
        """)

    return demo


if __name__ == "__main__":
    # Load predictor at startup
    print("Loading model...")
    try:
        predictor = load_predictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Running with randomly initialized weights")

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
