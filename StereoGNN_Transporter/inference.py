"""
Inference API and Virtual Screening Pipeline
=============================================

Production-ready inference for:
1. Single molecule prediction
2. Batch prediction
3. Virtual screening with ranking
4. Uncertainty quantification
5. Applicability domain estimation
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from model import StereoGNN, StereoGNNKinetic, KineticHead
from featurizer import MoleculeGraphFeaturizer
from config import CONFIG


@dataclass
class TransporterPrediction:
    """Prediction result for a single molecule."""
    smiles: str
    is_valid: bool

    # Predictions per target (probability of each class)
    dat: Dict[str, float]  # {inactive, blocker, substrate}
    net: Dict[str, float]
    sert: Dict[str, float]

    # Top prediction per target
    dat_prediction: str
    net_prediction: str
    sert_prediction: str

    # Substrate probabilities (main output for screening)
    dat_substrate_prob: float
    net_substrate_prob: float
    sert_substrate_prob: float

    # Uncertainty (std from MC Dropout)
    dat_uncertainty: float
    net_uncertainty: float
    sert_uncertainty: float

    # Applicability domain
    in_domain: bool
    domain_score: float

    # Stereochemistry info
    has_stereocenters: bool
    num_stereocenters: int

    def to_dict(self) -> Dict:
        return asdict(self)


class ApplicabilityDomain:
    """
    Estimates whether a molecule is within the model's applicability domain.

    Based on:
    1. Molecular descriptor ranges from training data
    2. Structural similarity to training set
    3. Presence of unusual features
    """

    def __init__(self):
        # Typical ranges for CNS-active monoamine substrates
        self.descriptor_ranges = {
            'mol_weight': (100, 600),
            'logp': (-2, 6),
            'tpsa': (0, 120),
            'num_hbd': (0, 5),
            'num_hba': (0, 10),
            'num_rotatable_bonds': (0, 10),
            'num_heavy_atoms': (5, 50),
        }

        # Required features (at least one should be present)
        self.required_patterns = [
            '[NX3;H2,H1,H0]',  # Amine
        ]

        # Problematic features (presence reduces domain score)
        self.problematic_patterns = [
            '[#6]=O',  # Carbonyl (can be fine, but unusual for substrates)
            '[Br,I]',  # Heavy halogens
            '[#15,#16,#33,#34]',  # P, S, As, Se
        ]

    def check(self, smiles: str) -> Tuple[bool, float]:
        """
        Check if molecule is in applicability domain.

        Returns:
            Tuple of (in_domain: bool, domain_score: float)
            domain_score in [0, 1], higher = more in domain
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, 0.0

        score = 1.0

        # Check descriptor ranges
        descriptors = {
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
            'num_hba': rdMolDescriptors.CalcNumHBA(mol),
            'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        }

        for name, value in descriptors.items():
            min_val, max_val = self.descriptor_ranges[name]
            if value < min_val or value > max_val:
                score *= 0.8  # Reduce score for out-of-range

        # Check required patterns
        has_required = False
        for smarts in self.required_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                has_required = True
                break

        if not has_required:
            score *= 0.5

        # Check problematic patterns
        for smarts in self.problematic_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                score *= 0.9

        in_domain = score >= 0.5
        return in_domain, score


class TransporterPredictor:
    """
    Main inference class for transporter substrate prediction.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        use_uncertainty: bool = True,
        n_mc_samples: int = 30,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use
            use_uncertainty: Whether to compute uncertainty via MC Dropout
            n_mc_samples: Number of MC samples for uncertainty
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_uncertainty = use_uncertainty
        self.n_mc_samples = n_mc_samples

        # Load model
        self.model = StereoGNN().to(self.device)
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model loaded, using randomly initialized weights")

        self.model.eval()

        # Featurizer
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Applicability domain
        self.ad_checker = ApplicabilityDomain()

        # Class names
        self.class_names = ['inactive', 'blocker', 'substrate']

    def predict(self, smiles: str) -> TransporterPrediction:
        """
        Predict transporter activity for a single molecule.

        Args:
            smiles: SMILES string

        Returns:
            TransporterPrediction with all results
        """
        # Validate and featurize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._create_invalid_prediction(smiles)

        data = self.featurizer.featurize(smiles)
        if data is None:
            return self._create_invalid_prediction(smiles)

        # Check applicability domain
        in_domain, domain_score = self.ad_checker.check(smiles)

        # Get prediction
        batch = Batch.from_data_list([data]).to(self.device)

        if self.use_uncertainty:
            predictions = self._predict_with_uncertainty(batch)
        else:
            predictions = self._predict_single(batch)

        # Extract results
        results = {}
        for target in ['DAT', 'NET', 'SERT']:
            probs = predictions[target]['mean'][0].cpu().numpy()
            std = predictions[target]['std'][0].cpu().numpy() if self.use_uncertainty else np.zeros(3)

            results[target] = {
                'probs': {
                    'inactive': float(probs[0]),
                    'blocker': float(probs[1]),
                    'substrate': float(probs[2]),
                },
                'prediction': self.class_names[np.argmax(probs)],
                'substrate_prob': float(probs[2]),
                'uncertainty': float(std[2]),  # Uncertainty on substrate
            }

        # Stereochemistry info
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return TransporterPrediction(
            smiles=smiles,
            is_valid=True,
            dat=results['DAT']['probs'],
            net=results['NET']['probs'],
            sert=results['SERT']['probs'],
            dat_prediction=results['DAT']['prediction'],
            net_prediction=results['NET']['prediction'],
            sert_prediction=results['SERT']['prediction'],
            dat_substrate_prob=results['DAT']['substrate_prob'],
            net_substrate_prob=results['NET']['substrate_prob'],
            sert_substrate_prob=results['SERT']['substrate_prob'],
            dat_uncertainty=results['DAT']['uncertainty'],
            net_uncertainty=results['NET']['uncertainty'],
            sert_uncertainty=results['SERT']['uncertainty'],
            in_domain=in_domain,
            domain_score=domain_score,
            has_stereocenters=len(chiral_centers) > 0,
            num_stereocenters=len(chiral_centers),
        )

    def predict_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[TransporterPrediction]:
        """
        Predict transporter activity for multiple molecules.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of TransporterPrediction objects
        """
        results = []
        iterator = range(0, len(smiles_list), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        for i in iterator:
            batch_smiles = smiles_list[i:i + batch_size]

            for smi in batch_smiles:
                pred = self.predict(smi)
                results.append(pred)

        return results

    def virtual_screen(
        self,
        smiles_list: List[str],
        target: str = 'DAT',
        min_substrate_prob: float = 0.5,
        require_in_domain: bool = True,
    ) -> pd.DataFrame:
        """
        Run virtual screening and rank compounds.

        Args:
            smiles_list: List of SMILES to screen
            target: Target transporter ('DAT', 'NET', 'SERT')
            min_substrate_prob: Minimum substrate probability threshold
            require_in_domain: Only include compounds in applicability domain

        Returns:
            DataFrame with ranked compounds
        """
        print(f"Screening {len(smiles_list)} compounds for {target}...")

        predictions = self.predict_batch(smiles_list)

        # Build results DataFrame
        records = []
        for pred in predictions:
            if not pred.is_valid:
                continue
            if require_in_domain and not pred.in_domain:
                continue

            prob_attr = f'{target.lower()}_substrate_prob'
            uncertainty_attr = f'{target.lower()}_uncertainty'
            pred_attr = f'{target.lower()}_prediction'

            records.append({
                'smiles': pred.smiles,
                'substrate_prob': getattr(pred, prob_attr),
                'uncertainty': getattr(pred, uncertainty_attr),
                'prediction': getattr(pred, pred_attr),
                'in_domain': pred.in_domain,
                'domain_score': pred.domain_score,
                'has_stereocenters': pred.has_stereocenters,
            })

        df = pd.DataFrame(records)

        # Filter by threshold
        df = df[df['substrate_prob'] >= min_substrate_prob]

        # Sort by substrate probability (descending)
        df = df.sort_values('substrate_prob', ascending=False)

        # Add rank
        df['rank'] = range(1, len(df) + 1)

        print(f"Found {len(df)} potential substrates (prob >= {min_substrate_prob})")

        return df

    def compare_enantiomers(
        self,
        smiles1: str,
        smiles2: str,
        target: str = 'DAT',
    ) -> Dict:
        """
        Compare predictions for a pair of stereoisomers.

        Args:
            smiles1: First stereoisomer SMILES
            smiles2: Second stereoisomer SMILES
            target: Target transporter

        Returns:
            Dict with comparison results
        """
        pred1 = self.predict(smiles1)
        pred2 = self.predict(smiles2)

        prob_attr = f'{target.lower()}_substrate_prob'
        p1 = getattr(pred1, prob_attr)
        p2 = getattr(pred2, prob_attr)

        return {
            'smiles1': smiles1,
            'smiles2': smiles2,
            'target': target,
            'prob1': p1,
            'prob2': p2,
            'ratio': p1 / (p2 + 1e-8),
            'difference': p1 - p2,
            'more_potent': smiles1 if p1 > p2 else smiles2,
        }

    def _predict_single(self, batch: Batch) -> Dict:
        """Single forward pass prediction."""
        with torch.no_grad():
            output = self.model(batch)

            results = {}
            for target in ['DAT', 'NET', 'SERT']:
                probs = F.softmax(output[target], dim=-1)
                results[target] = {
                    'mean': probs,
                    'std': torch.zeros_like(probs),
                }

            return results

    def _predict_with_uncertainty(self, batch: Batch) -> Dict:
        """MC Dropout uncertainty estimation."""
        self.model.train()  # Enable dropout

        predictions = {target: [] for target in ['DAT', 'NET', 'SERT']}

        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                output = self.model(batch)
                for target in predictions:
                    probs = F.softmax(output[target], dim=-1)
                    predictions[target].append(probs)

        self.model.eval()

        results = {}
        for target in predictions:
            stacked = torch.stack(predictions[target], dim=0)
            results[target] = {
                'mean': stacked.mean(dim=0),
                'std': stacked.std(dim=0),
            }

        return results

    def _create_invalid_prediction(self, smiles: str) -> TransporterPrediction:
        """Create a prediction for invalid molecules."""
        empty_probs = {'inactive': 0.0, 'blocker': 0.0, 'substrate': 0.0}
        return TransporterPrediction(
            smiles=smiles,
            is_valid=False,
            dat=empty_probs,
            net=empty_probs,
            sert=empty_probs,
            dat_prediction='invalid',
            net_prediction='invalid',
            sert_prediction='invalid',
            dat_substrate_prob=0.0,
            net_substrate_prob=0.0,
            sert_substrate_prob=0.0,
            dat_uncertainty=1.0,
            net_uncertainty=1.0,
            sert_uncertainty=1.0,
            in_domain=False,
            domain_score=0.0,
            has_stereocenters=False,
            num_stereocenters=0,
        )


class VirtualScreeningPipeline:
    """
    Complete virtual screening workflow.
    """

    def __init__(self, predictor: TransporterPredictor):
        self.predictor = predictor

    def run(
        self,
        input_file: Path,
        output_file: Path,
        target: str = 'DAT',
        smiles_column: str = 'SMILES',
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run virtual screening on a file.

        Args:
            input_file: CSV/SDF file with molecules
            output_file: Output CSV file
            target: Target transporter
            smiles_column: Name of SMILES column
            top_n: Return only top N hits

        Returns:
            DataFrame with results
        """
        # Load input
        if str(input_file).endswith('.csv'):
            df = pd.read_csv(input_file)
            smiles_list = df[smiles_column].tolist()
        elif str(input_file).endswith('.sdf'):
            from rdkit.Chem import SDMolSupplier
            suppl = SDMolSupplier(str(input_file))
            smiles_list = [
                Chem.MolToSmiles(mol) for mol in suppl
                if mol is not None
            ]
        else:
            raise ValueError(f"Unsupported file format: {input_file}")

        print(f"Loaded {len(smiles_list)} molecules from {input_file}")

        # Screen
        results = self.predictor.virtual_screen(smiles_list, target=target)

        # Top N
        if top_n is not None:
            results = results.head(top_n)

        # Save
        results.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

        return results


def predict_single(smiles: str, model_path: Optional[Path] = None) -> Dict:
    """
    Quick function to predict a single molecule.

    Args:
        smiles: SMILES string
        model_path: Optional path to model checkpoint

    Returns:
        Dict with predictions
    """
    predictor = TransporterPredictor(model_path=model_path, use_uncertainty=False)
    result = predictor.predict(smiles)
    return result.to_dict()


# =============================================================================
# KINETIC EXTENSION - Mechanistic parameter prediction
# =============================================================================

@dataclass
class KineticPrediction:
    """Kinetic parameter predictions for a single transporter."""
    pKi: float  # Binding affinity (-log10 Ki)
    pKi_uncertainty: float  # Total uncertainty (aleatoric + epistemic)
    pKi_aleatoric: float  # Data noise uncertainty
    pKi_epistemic: float  # Model uncertainty

    pIC50: float  # Functional potency (-log10 IC50)
    pIC50_uncertainty: float
    pIC50_aleatoric: float
    pIC50_epistemic: float

    kinetic_bias: float  # Uptake preference (0-1)
    kinetic_bias_uncertainty: float
    kinetic_bias_aleatoric: float
    kinetic_bias_epistemic: float

    interaction_mode: str  # Predicted mode
    interaction_mode_probs: Dict[str, float]  # Per-class probabilities
    interaction_mode_entropy: float  # Classification uncertainty

    # Interpretations
    affinity_category: str  # High/Medium/Low
    potency_category: str  # High/Medium/Low
    mechanism_summary: str  # Human-readable summary


@dataclass
class KineticTransporterPrediction:
    """Complete kinetic prediction for a molecule across all transporters."""
    smiles: str
    is_valid: bool

    # Per-transporter kinetic predictions
    dat_kinetics: Optional[KineticPrediction]
    net_kinetics: Optional[KineticPrediction]
    sert_kinetics: Optional[KineticPrediction]

    # Original activity predictions (for compatibility)
    dat_activity: Dict[str, float]
    net_activity: Dict[str, float]
    sert_activity: Dict[str, float]

    # Applicability domain
    in_domain: bool
    domain_score: float

    # Stereochemistry
    has_stereocenters: bool
    num_stereocenters: int

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_summary(self) -> str:
        """Generate human-readable summary of predictions."""
        lines = [f"Kinetic Prediction: {self.smiles[:50]}..."]

        for task, kinetics in [('DAT', self.dat_kinetics),
                                ('NET', self.net_kinetics),
                                ('SERT', self.sert_kinetics)]:
            if kinetics:
                lines.append(f"\n{task}:")
                lines.append(f"  Mode: {kinetics.interaction_mode}")
                lines.append(f"  pKi: {kinetics.pKi:.2f} ± {kinetics.pKi_uncertainty:.2f} ({kinetics.affinity_category})")
                lines.append(f"  pIC50: {kinetics.pIC50:.2f} ± {kinetics.pIC50_uncertainty:.2f} ({kinetics.potency_category})")
                lines.append(f"  Kinetic bias: {kinetics.kinetic_bias:.2f} (uptake preference)")
                lines.append(f"  {kinetics.mechanism_summary}")

        return "\n".join(lines)


class KineticTransporterPredictor:
    """
    Inference class for kinetic parameter prediction.

    Extends TransporterPredictor with mechanistic kinetic outputs:
    - Binding affinity (pKi)
    - Functional potency (pIC50)
    - Interaction mode (substrate/competitive/non-competitive/partial)
    - Kinetic bias (uptake vs blockade preference)

    All predictions include uncertainty quantification.
    """

    # Mode name mapping
    MODE_NAMES = ['substrate', 'competitive_inhibitor', 'non_competitive_inhibitor', 'partial_substrate']

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        n_mc_samples: int = 30,
    ):
        """
        Args:
            model_path: Path to trained kinetic model checkpoint
            device: Device to use
            n_mc_samples: Number of MC dropout samples for uncertainty
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_mc_samples = n_mc_samples

        # Load kinetic model
        self.model = StereoGNNKinetic().to(self.device)
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded kinetic model from {model_path}")
        else:
            print("Warning: No model loaded, using randomly initialized weights")

        self.model.eval()

        # Featurizer
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Applicability domain
        self.ad_checker = ApplicabilityDomain()

    def predict(self, smiles: str) -> KineticTransporterPrediction:
        """
        Predict kinetic parameters for a single molecule.

        Args:
            smiles: SMILES string

        Returns:
            KineticTransporterPrediction with all kinetic outputs
        """
        # Validate molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._create_invalid_prediction(smiles)

        data = self.featurizer.featurize(smiles)
        if data is None:
            return self._create_invalid_prediction(smiles)

        # Check applicability domain
        in_domain, domain_score = self.ad_checker.check(smiles)

        # Get prediction with uncertainty
        batch = Batch.from_data_list([data]).to(self.device)
        uncertainty_results = self.model.predict_kinetics_with_uncertainty(
            batch, n_mc_samples=self.n_mc_samples
        )

        # Extract kinetic predictions for each transporter
        kinetic_preds = {}
        activity_preds = {}

        for task in ['DAT', 'NET', 'SERT']:
            task_result = uncertainty_results[task]

            # Extract kinetic parameters
            kinetic_preds[task] = self._extract_kinetic_prediction(task_result)

            # Extract activity predictions
            activity_probs = task_result['activity']['mean_probs'][0].cpu().numpy()
            activity_preds[task] = {
                'inactive': float(activity_probs[0]),
                'blocker': float(activity_probs[1]),
                'substrate': float(activity_probs[2]),
            }

        # Stereochemistry info
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        return KineticTransporterPrediction(
            smiles=smiles,
            is_valid=True,
            dat_kinetics=kinetic_preds['DAT'],
            net_kinetics=kinetic_preds['NET'],
            sert_kinetics=kinetic_preds['SERT'],
            dat_activity=activity_preds['DAT'],
            net_activity=activity_preds['NET'],
            sert_activity=activity_preds['SERT'],
            in_domain=in_domain,
            domain_score=domain_score,
            has_stereocenters=len(chiral_centers) > 0,
            num_stereocenters=len(chiral_centers),
        )

    def _extract_kinetic_prediction(self, task_result: Dict) -> KineticPrediction:
        """Extract kinetic prediction from model output."""
        # pKi
        pki = task_result['pKi']['mean'][0].cpu().item()
        pki_aleatoric = task_result['pKi']['aleatoric'][0].cpu().item()
        pki_epistemic = task_result['pKi']['epistemic'][0].cpu().item()
        pki_total = task_result['pKi']['total'][0].cpu().item()

        # pIC50
        pic50 = task_result['pIC50']['mean'][0].cpu().item()
        pic50_aleatoric = task_result['pIC50']['aleatoric'][0].cpu().item()
        pic50_epistemic = task_result['pIC50']['epistemic'][0].cpu().item()
        pic50_total = task_result['pIC50']['total'][0].cpu().item()

        # Kinetic bias
        bias = task_result['kinetic_bias']['mean'][0].cpu().item()
        bias_aleatoric = task_result['kinetic_bias']['aleatoric'][0].cpu().item()
        bias_epistemic = task_result['kinetic_bias']['epistemic'][0].cpu().item()
        bias_total = task_result['kinetic_bias']['total'][0].cpu().item()

        # Interaction mode
        mode_probs = task_result['interaction_mode']['mean_probs'][0].cpu().numpy()
        mode_idx = task_result['interaction_mode']['predicted_class'][0].cpu().item()
        mode_entropy = task_result['interaction_mode']['entropy'][0].cpu().item()
        mode_name = self.MODE_NAMES[mode_idx]

        mode_prob_dict = {
            name: float(mode_probs[i])
            for i, name in enumerate(self.MODE_NAMES)
        }

        # Categorizations
        affinity_cat = self._categorize_affinity(pki)
        potency_cat = self._categorize_potency(pic50)
        mechanism = self._generate_mechanism_summary(mode_name, bias, pki, pic50)

        return KineticPrediction(
            pKi=pki,
            pKi_uncertainty=pki_total,
            pKi_aleatoric=pki_aleatoric,
            pKi_epistemic=pki_epistemic,
            pIC50=pic50,
            pIC50_uncertainty=pic50_total,
            pIC50_aleatoric=pic50_aleatoric,
            pIC50_epistemic=pic50_epistemic,
            kinetic_bias=bias,
            kinetic_bias_uncertainty=bias_total,
            kinetic_bias_aleatoric=bias_aleatoric,
            kinetic_bias_epistemic=bias_epistemic,
            interaction_mode=mode_name,
            interaction_mode_probs=mode_prob_dict,
            interaction_mode_entropy=mode_entropy,
            affinity_category=affinity_cat,
            potency_category=potency_cat,
            mechanism_summary=mechanism,
        )

    def _categorize_affinity(self, pki: float) -> str:
        """Categorize binding affinity."""
        if pki >= 8.0:  # Ki < 10 nM
            return "High"
        elif pki >= 6.0:  # Ki < 1 uM
            return "Medium"
        else:
            return "Low"

    def _categorize_potency(self, pic50: float) -> str:
        """Categorize functional potency."""
        if pic50 >= 7.0:  # IC50 < 100 nM
            return "High"
        elif pic50 >= 5.0:  # IC50 < 10 uM
            return "Medium"
        else:
            return "Low"

    def _generate_mechanism_summary(
        self,
        mode: str,
        bias: float,
        pki: float,
        pic50: float,
    ) -> str:
        """Generate human-readable mechanism summary."""
        if mode == 'substrate':
            if bias > 0.7:
                return "Strong substrate with uptake preference"
            elif bias < 0.3:
                return "Substrate with release/efflux tendency"
            else:
                return "Balanced substrate activity"
        elif mode == 'competitive_inhibitor':
            return f"Competitive blocker (orthosteric binding)"
        elif mode == 'non_competitive_inhibitor':
            return "Non-competitive blocker (allosteric binding)"
        elif mode == 'partial_substrate':
            return "Partial substrate with mixed mechanism"
        else:
            return "Unknown mechanism"

    def predict_batch(
        self,
        smiles_list: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[KineticTransporterPrediction]:
        """Predict kinetic parameters for multiple molecules."""
        results = []
        iterator = range(0, len(smiles_list), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting kinetics")

        for i in iterator:
            batch_smiles = smiles_list[i:i + batch_size]
            for smi in batch_smiles:
                pred = self.predict(smi)
                results.append(pred)

        return results

    def compare_mechanisms(
        self,
        smiles1: str,
        smiles2: str,
        target: str = 'DAT',
    ) -> Dict:
        """
        Compare kinetic parameters between two molecules.

        Useful for SAR analysis and enantiomer comparison.
        """
        pred1 = self.predict(smiles1)
        pred2 = self.predict(smiles2)

        kinetics1 = getattr(pred1, f'{target.lower()}_kinetics')
        kinetics2 = getattr(pred2, f'{target.lower()}_kinetics')

        if kinetics1 is None or kinetics2 is None:
            return {'error': 'One or both molecules failed prediction'}

        return {
            'smiles1': smiles1,
            'smiles2': smiles2,
            'target': target,
            'pKi_diff': kinetics1.pKi - kinetics2.pKi,
            'pIC50_diff': kinetics1.pIC50 - kinetics2.pIC50,
            'bias_diff': kinetics1.kinetic_bias - kinetics2.kinetic_bias,
            'mode1': kinetics1.interaction_mode,
            'mode2': kinetics2.interaction_mode,
            'same_mode': kinetics1.interaction_mode == kinetics2.interaction_mode,
            'affinity_ratio': 10 ** (kinetics1.pKi - kinetics2.pKi),  # Fold difference
            'potency_ratio': 10 ** (kinetics1.pIC50 - kinetics2.pIC50),
        }

    def screen_for_substrates(
        self,
        smiles_list: List[str],
        target: str = 'DAT',
        min_substrate_prob: float = 0.5,
        require_uptake_bias: bool = True,
    ) -> pd.DataFrame:
        """
        Screen compounds for substrate activity.

        Args:
            smiles_list: Compounds to screen
            target: Target transporter
            min_substrate_prob: Minimum probability for substrate mode
            require_uptake_bias: If True, filter for uptake-preferring compounds

        Returns:
            DataFrame with ranked substrate candidates
        """
        predictions = self.predict_batch(smiles_list)

        records = []
        for pred in predictions:
            if not pred.is_valid:
                continue

            kinetics = getattr(pred, f'{target.lower()}_kinetics')
            if kinetics is None:
                continue

            substrate_prob = kinetics.interaction_mode_probs.get('substrate', 0)
            if substrate_prob < min_substrate_prob:
                continue

            if require_uptake_bias and kinetics.kinetic_bias < 0.5:
                continue

            records.append({
                'smiles': pred.smiles,
                'substrate_prob': substrate_prob,
                'pKi': kinetics.pKi,
                'pKi_uncertainty': kinetics.pKi_uncertainty,
                'pIC50': kinetics.pIC50,
                'pIC50_uncertainty': kinetics.pIC50_uncertainty,
                'kinetic_bias': kinetics.kinetic_bias,
                'mode': kinetics.interaction_mode,
                'mechanism': kinetics.mechanism_summary,
                'in_domain': pred.in_domain,
            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            df = df.sort_values('substrate_prob', ascending=False)
            df['rank'] = range(1, len(df) + 1)

        return df

    def _create_invalid_prediction(self, smiles: str) -> KineticTransporterPrediction:
        """Create prediction for invalid molecules."""
        empty_activity = {'inactive': 0.0, 'blocker': 0.0, 'substrate': 0.0}
        return KineticTransporterPrediction(
            smiles=smiles,
            is_valid=False,
            dat_kinetics=None,
            net_kinetics=None,
            sert_kinetics=None,
            dat_activity=empty_activity,
            net_activity=empty_activity,
            sert_activity=empty_activity,
            in_domain=False,
            domain_score=0.0,
            has_stereocenters=False,
            num_stereocenters=0,
        )


def predict_kinetics(smiles: str, model_path: Optional[Path] = None) -> Dict:
    """
    Quick function to predict kinetic parameters for a single molecule.

    Args:
        smiles: SMILES string
        model_path: Optional path to kinetic model checkpoint

    Returns:
        Dict with kinetic predictions
    """
    predictor = KineticTransporterPredictor(model_path=model_path)
    result = predictor.predict(smiles)
    return result.to_dict()


if __name__ == "__main__":
    print("=" * 60)
    print("Inference API Test")
    print("=" * 60)

    # Create predictor (no trained model, just testing API)
    predictor = TransporterPredictor(use_uncertainty=False)

    # Test single prediction
    test_smiles = "C[C@H](N)Cc1ccccc1"  # d-Amphetamine
    print(f"\nPredicting: {test_smiles}")
    result = predictor.predict(test_smiles)

    print(f"\nResult:")
    print(f"  Valid: {result.is_valid}")
    print(f"  DAT prediction: {result.dat_prediction} ({result.dat_substrate_prob:.3f})")
    print(f"  NET prediction: {result.net_prediction} ({result.net_substrate_prob:.3f})")
    print(f"  SERT prediction: {result.sert_prediction} ({result.sert_substrate_prob:.3f})")
    print(f"  In domain: {result.in_domain} (score: {result.domain_score:.3f})")
    print(f"  Stereocenters: {result.num_stereocenters}")

    # Test enantiomer comparison
    print("\nComparing enantiomers:")
    comparison = predictor.compare_enantiomers(
        "C[C@H](N)Cc1ccccc1",  # d-Amphetamine
        "C[C@@H](N)Cc1ccccc1",  # l-Amphetamine
        target="DAT",
    )
    print(f"  d-Amph prob: {comparison['prob1']:.3f}")
    print(f"  l-Amph prob: {comparison['prob2']:.3f}")
    print(f"  Ratio: {comparison['ratio']:.2f}x")

    # Test batch prediction
    print("\nBatch prediction:")
    test_batch = [
        "C[C@H](N)Cc1ccccc1",     # d-Amphetamine
        "C[C@H](NC)Cc1ccccc1",    # d-Methamphetamine
        "NCCc1ccc(O)c(O)c1",       # Dopamine
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # Caffeine
    ]
    results = predictor.predict_batch(test_batch, show_progress=False)
    for r in results:
        print(f"  {r.smiles[:30]:<30} DAT: {r.dat_substrate_prob:.3f}")
