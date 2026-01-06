"""
StereoGNN Transporter Substrate Predictor
==========================================

A stereochemistry-aware graph neural network for predicting substrate-likeness
of small molecules for monoamine transporters (DAT, NET, SERT).

Key Features:
- Explicit stereochemistry encoding in node/edge features
- Multi-task learning for DAT/NET/SERT
- Rigorous substrate vs blocker distinction
- MC Dropout uncertainty quantification
- Interpretability via attention and integrated gradients

Usage:
    from stereo_gnn import TransporterPredictor

    predictor = TransporterPredictor(model_path="models/best_model.pt")
    result = predictor.predict("C[C@H](N)Cc1ccccc1")  # d-Amphetamine

    print(f"DAT substrate probability: {result.dat_substrate_prob:.3f}")
"""

from .model import StereoGNN, StereoGNNForAblation
from .inference import TransporterPredictor, TransporterPrediction
from .featurizer import MoleculeGraphFeaturizer
from .config import CONFIG

__version__ = "1.0.0"
__author__ = "StereoGNN Team"

__all__ = [
    "StereoGNN",
    "StereoGNNForAblation",
    "TransporterPredictor",
    "TransporterPrediction",
    "MoleculeGraphFeaturizer",
    "CONFIG",
]
