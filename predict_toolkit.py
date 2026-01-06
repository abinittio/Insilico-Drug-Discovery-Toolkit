"""
Insilico Drug Discovery Toolkit - Unified Predictor
====================================================
Combines all trained models into one inference pipeline.

Modules:
- BBB Permeability
- MAT Kinetics (DAT/NET/SERT)
- Abuse Liability
- hERG Cardiotoxicity
- CYP450 Metabolism

Usage:
    from predict_toolkit import DrugDiscoveryToolkit

    toolkit = DrugDiscoveryToolkit()
    results = toolkit.predict("CCO")  # Ethanol
    print(results)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from model import StereoGNN
from featurizer import MoleculeGraphFeaturizer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path(__file__).parent / 'models'


@dataclass
class PredictionResult:
    """Container for all predictions."""
    smiles: str
    valid: bool = True
    error: str = None

    # BBB
    bbb_permeable: bool = None
    bbb_score: float = None

    # MAT Kinetics
    dat_class: str = None  # substrate/blocker/inactive
    dat_pki: float = None
    net_class: str = None
    net_pki: float = None
    sert_class: str = None
    sert_pki: float = None

    # Abuse Liability
    abuse_score: float = None
    abuse_category: str = None  # low/moderate/high

    # hERG Cardiotoxicity
    herg_active: bool = None
    herg_prob: float = None
    herg_pic50: float = None
    qt_risk: str = None  # low/moderate/high

    # CYP Metabolism
    cyp1a2_inhibitor: bool = None
    cyp2c9_inhibitor: bool = None
    cyp2c19_inhibitor: bool = None
    cyp2d6_inhibitor: bool = None
    cyp3a4_inhibitor: bool = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"=== Prediction for {self.smiles} ===\n"]

        if not self.valid:
            return f"Invalid molecule: {self.error}"

        # BBB
        if self.bbb_score is not None:
            bbb_status = "YES" if self.bbb_permeable else "NO"
            lines.append(f"BBB Permeable: {bbb_status} ({self.bbb_score:.2f})")

        # MAT
        if self.dat_class:
            lines.append(f"\nMAT Kinetics:")
            lines.append(f"  DAT: {self.dat_class} (pKi: {self.dat_pki:.2f})" if self.dat_pki else f"  DAT: {self.dat_class}")
            lines.append(f"  NET: {self.net_class} (pKi: {self.net_pki:.2f})" if self.net_pki else f"  NET: {self.net_class}")
            lines.append(f"  SERT: {self.sert_class} (pKi: {self.sert_pki:.2f})" if self.sert_pki else f"  SERT: {self.sert_class}")

        # Abuse
        if self.abuse_score is not None:
            lines.append(f"\nAbuse Liability: {self.abuse_category.upper()} ({self.abuse_score:.0f}/100)")

        # hERG
        if self.herg_prob is not None:
            herg_status = "BLOCKER" if self.herg_active else "SAFE"
            lines.append(f"\nhERG Cardiotoxicity: {herg_status} (prob: {self.herg_prob:.2f})")
            lines.append(f"  QT Prolongation Risk: {self.qt_risk}")

        # CYP
        cyp_inhibitors = []
        if self.cyp1a2_inhibitor: cyp_inhibitors.append("1A2")
        if self.cyp2c9_inhibitor: cyp_inhibitors.append("2C9")
        if self.cyp2c19_inhibitor: cyp_inhibitors.append("2C19")
        if self.cyp2d6_inhibitor: cyp_inhibitors.append("2D6")
        if self.cyp3a4_inhibitor: cyp_inhibitors.append("3A4")

        if cyp_inhibitors:
            lines.append(f"\nCYP Inhibition: {', '.join(cyp_inhibitors)}")
        else:
            lines.append(f"\nCYP Inhibition: None predicted")

        return '\n'.join(lines)


def get_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    """Generate Morgan fingerprint + descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    fp_arr = np.zeros(n_bits)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolWt(mol) / 500,
        Descriptors.NumHDonors(mol) / 5,
        Descriptors.NumHAcceptors(mol) / 10,
        Descriptors.NumRotatableBonds(mol) / 10,
        Descriptors.NumAromaticRings(mol) / 5,
        Descriptors.FractionCSP3(mol),
    ]

    return np.concatenate([fp_arr, desc]).astype(np.float32)


class DrugDiscoveryToolkit:
    """Unified prediction interface for all models."""

    def __init__(self, model_dir: Path = None, device: torch.device = None):
        self.model_dir = model_dir or MODEL_DIR
        self.device = device or DEVICE
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Lazy load models
        self._herg_model = None
        self._cyp_model = None
        self._mat_model = None

        print(f"DrugDiscoveryToolkit initialized (device: {self.device})")
        self._check_models()

    def _check_models(self):
        """Check which models are available."""
        models = {
            'hERG': self.model_dir / 'herg' / 'best_fold0.pt',
            'CYP': self.model_dir / 'cyp' / 'best_cyp_model.pt',
            'MAT': self.model_dir / 'kinetic_v3' / 'best_kinetic_model.pt',
        }

        print("Available models:")
        for name, path in models.items():
            status = "✓" if path.exists() else "✗"
            print(f"  {status} {name}: {path}")

    def _load_herg(self):
        """Load hERG model."""
        if self._herg_model is not None:
            return

        # Import here to avoid circular imports
        from train_herg import HERGClassifier, FP_DIM

        path = self.model_dir / 'herg' / 'best_fold0.pt'
        if not path.exists():
            print("hERG model not found")
            return

        backbone = StereoGNN()
        self._herg_model = HERGClassifier(backbone, fp_dim=FP_DIM).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._herg_model.load_state_dict(ckpt['model_state_dict'])
        self._herg_model.eval()
        print("hERG model loaded")

    def _load_cyp(self):
        """Load CYP model."""
        if self._cyp_model is not None:
            return

        from train_cyp import CYPClassifier, CYP_TARGETS

        path = self.model_dir / 'cyp' / 'best_cyp_model.pt'
        if not path.exists():
            print("CYP model not found")
            return

        backbone = StereoGNN()
        self._cyp_model = CYPClassifier(backbone, n_targets=len(CYP_TARGETS)).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._cyp_model.load_state_dict(ckpt['model_state_dict'])
        self._cyp_model.eval()
        print("CYP model loaded")

    @torch.no_grad()
    def predict(self, smiles: str) -> PredictionResult:
        """Run all predictions on a molecule."""
        result = PredictionResult(smiles=smiles)

        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result.valid = False
            result.error = "Invalid SMILES"
            return result

        # Create graph
        try:
            data = self.featurizer.featurize(smiles, {'DAT': -1, 'NET': -1, 'SERT': -1})
            if data is None:
                result.valid = False
                result.error = "Featurization failed"
                return result

            fp = get_fingerprint(smiles)
            if fp is not None:
                data.fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)

            data = data.to(self.device)
        except Exception as e:
            result.valid = False
            result.error = str(e)
            return result

        # hERG prediction
        try:
            self._load_herg()
            if self._herg_model is not None:
                from torch_geometric.data import Batch
                batch = Batch.from_data_list([data])
                output = self._herg_model(batch)

                prob = torch.softmax(output['logits'], dim=-1)[0, 1].item()
                result.herg_prob = prob
                result.herg_active = prob > 0.5
                result.herg_pic50 = output['pIC50'][0].item()

                # QT risk based on probability
                if prob < 0.3:
                    result.qt_risk = "low"
                elif prob < 0.7:
                    result.qt_risk = "moderate"
                else:
                    result.qt_risk = "high"
        except Exception as e:
            print(f"hERG prediction failed: {e}")

        # CYP prediction
        try:
            self._load_cyp()
            if self._cyp_model is not None:
                from torch_geometric.data import Batch
                from train_cyp import CYP_TARGETS

                batch = Batch.from_data_list([data])
                batch.mask = torch.ones(1, len(CYP_TARGETS))
                batch.y = torch.zeros(1, len(CYP_TARGETS), dtype=torch.long)
                output = self._cyp_model(batch)

                for i, cyp in enumerate(CYP_TARGETS):
                    prob = torch.softmax(output[cyp], dim=-1)[0, 1].item()
                    is_inhibitor = prob > 0.5
                    setattr(result, f'{cyp.lower()}_inhibitor', is_inhibitor)
        except Exception as e:
            print(f"CYP prediction failed: {e}")

        return result

    def predict_batch(self, smiles_list: List[str]) -> List[PredictionResult]:
        """Predict for multiple molecules."""
        return [self.predict(s) for s in smiles_list]

    def predict_csv(self, input_path: str, output_path: str, smiles_col: str = 'smiles'):
        """Predict from CSV file."""
        import pandas as pd

        df = pd.read_csv(input_path)
        results = self.predict_batch(df[smiles_col].tolist())

        # Convert to DataFrame
        result_dicts = [r.to_dict() for r in results]
        result_df = pd.DataFrame(result_dicts)

        # Merge with original
        output_df = pd.concat([df, result_df.drop('smiles', axis=1)], axis=1)
        output_df.to_csv(output_path, index=False)

        print(f"Predictions saved to {output_path}")
        return output_df


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Drug Discovery Toolkit')
    parser.add_argument('smiles', nargs='?', help='SMILES string to predict')
    parser.add_argument('--csv', help='Input CSV file')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--smiles-col', default='smiles', help='SMILES column name')
    args = parser.parse_args()

    toolkit = DrugDiscoveryToolkit()

    if args.csv:
        if not args.output:
            args.output = args.csv.replace('.csv', '_predictions.csv')
        toolkit.predict_csv(args.csv, args.output, args.smiles_col)
    elif args.smiles:
        result = toolkit.predict(args.smiles)
        print(result.summary())
    else:
        # Demo
        demo_molecules = [
            ('Caffeine', 'Cn1cnc2c1c(=O)n(c(=O)n2C)C'),
            ('Cocaine', 'COC(=O)C1C2CCC(CC1OC(=O)C1=CC=CC=C1)N2C'),
            ('Amphetamine', 'CC(N)Cc1ccccc1'),
        ]

        print("=== Demo Predictions ===\n")
        for name, smi in demo_molecules:
            print(f"\n--- {name} ---")
            result = toolkit.predict(smi)
            print(result.summary())


if __name__ == "__main__":
    main()
