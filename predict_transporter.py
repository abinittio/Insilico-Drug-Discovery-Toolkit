"""
StereoGNN Transporter Substrate Predictor
==========================================

Production-ready interface for monoamine transporter substrate prediction.
Designed to match BBBGNNPredictor interface for future tool combination.

Targets:
- DAT (Dopamine Transporter)
- NET (Norepinephrine Transporter)
- SERT (Serotonin Transporter)

Usage:
    from predict_transporter import TransporterGNNPredictor

    predictor = TransporterGNNPredictor()
    result = predictor.predict("C[C@H](N)Cc1ccccc1")  # d-Amphetamine

    # Batch prediction
    results = predictor.predict_batch(["CCO", "c1ccccc1CCN"])
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from run_training import StereoGNNSmallFinetune
from featurizer import MoleculeGraphFeaturizer
from config import CONFIG


# Electronegativity values (Pauling scale) for common atoms
ELECTRONEGATIVITY = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}


def get_molecular_descriptors(smiles: str) -> Optional[Dict]:
    """
    Calculate molecular descriptors relevant for transporter substrate prediction.

    Returns dict with key descriptors including:
    - Basic physicochemical properties
    - CNS drug-likeness indicators
    - Transporter-relevant features
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)

    descriptors = {
        'molecular_weight': mw,
        'logp': logp,
        'tpsa': tpsa,
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
        'fraction_csp3': Descriptors.FractionCSP3(mol),
        'num_rings': Descriptors.RingCount(mol),
    }

    # Check for basic nitrogen (important for transporter substrates)
    has_basic_n = False
    num_basic_n = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:  # Nitrogen
            # Check if basic (primary/secondary amine or non-aromatic tertiary)
            if atom.GetTotalNumHs() > 0 or (atom.GetDegree() == 3 and not atom.GetIsAromatic()):
                # Exclude amides
                is_amide = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6:
                        for n_neighbor in neighbor.GetNeighbors():
                            if n_neighbor.GetAtomicNum() == 8:
                                bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx())
                                if bond and bond.GetBondTypeAsDouble() == 2.0:
                                    is_amide = True
                                    break
                if not is_amide:
                    has_basic_n = True
                    num_basic_n += 1

    descriptors['has_basic_nitrogen'] = has_basic_n
    descriptors['num_basic_nitrogens'] = num_basic_n

    # Stereochemistry info
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    descriptors['num_stereocenters'] = len(chiral_centers)
    descriptors['has_stereocenters'] = len(chiral_centers) > 0

    # CNS drug-likeness check (typical range for CNS-active compounds)
    cns_compliant = (
        100 <= mw <= 500 and
        -1 <= logp <= 5 and
        tpsa <= 90 and
        descriptors['num_h_donors'] <= 3 and
        descriptors['num_h_acceptors'] <= 7
    )
    descriptors['cns_drug_like'] = cns_compliant

    # Transporter substrate likelihood indicators
    # Most substrates have basic nitrogen and phenethylamine-like structure
    phenethylamine_pattern = Chem.MolFromSmarts('c1ccccc1CCN')
    amphetamine_pattern = Chem.MolFromSmarts('c1ccccc1CC(C)N')
    cathinone_pattern = Chem.MolFromSmarts('c1ccccc1C(=O)C(C)N')

    descriptors['is_phenethylamine'] = mol.HasSubstructMatch(phenethylamine_pattern) if phenethylamine_pattern else False
    descriptors['is_amphetamine_like'] = mol.HasSubstructMatch(amphetamine_pattern) if amphetamine_pattern else False
    descriptors['is_cathinone_like'] = mol.HasSubstructMatch(cathinone_pattern) if cathinone_pattern else False

    return descriptors


class TransporterGNNPredictor:
    """
    Production-ready monoamine transporter substrate predictor using trained GNN model.
    Uses StereoGNN architecture with stereochemistry-aware features.

    Interface designed to match BBBGNNPredictor for future tool combination.
    """

    def __init__(self, model_path: str = None, device=None):
        """
        Initialize the predictor with a trained model.

        Args:
            model_path: Path to saved model checkpoint. If None, uses default.
            device: torch device (auto-detects if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize StereoGNN model (fine-tuned for monoamine transporters)
        # Use small model architecture that was used during training
        self.model = StereoGNNSmallFinetune(node_dim=86, edge_dim=18).to(self.device)

        # Default model path
        if model_path is None:
            model_path = CONFIG.data.project_root / "outputs" / "best_model.pt"
        else:
            model_path = Path(model_path)

        # Load trained weights
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.trained = True
            print(f"Loaded trained model from {model_path}")
            if 'val_auroc' in checkpoint:
                print(f"  Validation AUROC: {checkpoint['val_auroc']:.4f}")
        else:
            self.trained = False
            print(f"Warning: Model file not found at {model_path}")
            print("Model initialized but not trained. Predictions will be random.")

        # Featurizer for converting SMILES to graphs
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Class names for interpretation
        self.class_names = ['inactive', 'blocker', 'substrate']
        self.targets = ['DAT', 'NET', 'SERT']

    def predict(self, smiles: str, return_details: bool = True) -> Dict:
        """
        Predict transporter activity for a molecule.

        Args:
            smiles: SMILES string of molecule
            return_details: If True, return detailed analysis including descriptors

        Returns:
            dict with predictions and optional details
        """
        if not self.trained:
            print("Warning: Using untrained model!")

        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'success': False,
                'error': 'Invalid SMILES string',
                'smiles': smiles
            }

        # Convert to graph
        data = self.featurizer.featurize(smiles)
        if data is None:
            return {
                'success': False,
                'error': 'Failed to featurize molecule',
                'smiles': smiles
            }

        # Create batch for single molecule
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data]).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch)

            # Apply softmax to get probabilities
            import torch.nn.functional as F
            results = {}
            for target in self.targets:
                probs = F.softmax(output[target], dim=-1)[0].cpu().numpy()
                results[target] = {
                    'inactive_prob': float(probs[0]),
                    'blocker_prob': float(probs[1]),
                    'substrate_prob': float(probs[2]),
                    'prediction': self.class_names[probs.argmax()],
                }

        # Build response
        response = {
            'success': True,
            'smiles': smiles,
        }

        # Add per-target results
        for target in self.targets:
            target_lower = target.lower()
            response[f'{target_lower}_score'] = results[target]['substrate_prob']
            response[f'{target_lower}_prediction'] = results[target]['prediction']
            response[f'{target_lower}_probs'] = results[target]

        # Overall interpretation
        interpretations = []
        for target in self.targets:
            pred = results[target]['prediction']
            prob = results[target]['substrate_prob']
            if pred == 'substrate':
                interpretations.append(f"{target} substrate (prob: {prob:.3f})")
            elif pred == 'blocker':
                interpretations.append(f"{target} blocker")
            else:
                interpretations.append(f"{target} inactive")

        response['interpretation'] = "; ".join(interpretations)

        # Categorize overall activity
        substrate_count = sum(1 for t in self.targets if results[t]['prediction'] == 'substrate')
        if substrate_count == 0:
            response['category'] = 'NON-SUBSTRATE'
        elif substrate_count == len(self.targets):
            response['category'] = 'PAN-SUBSTRATE'
        else:
            selective_targets = [t for t in self.targets if results[t]['prediction'] == 'substrate']
            response['category'] = f"SELECTIVE ({', '.join(selective_targets)})"

        if return_details:
            # Get molecular descriptors
            descriptors = get_molecular_descriptors(smiles)

            if descriptors:
                response['molecular_descriptors'] = descriptors
                response['cns_drug_like'] = descriptors['cns_drug_like']
                response['has_basic_nitrogen'] = descriptors['has_basic_nitrogen']
                response['has_stereocenters'] = descriptors['has_stereocenters']
                response['num_stereocenters'] = descriptors['num_stereocenters']

                # Add warnings if any
                warnings = []
                if not descriptors['has_basic_nitrogen']:
                    warnings.append('No basic nitrogen (unusual for substrates)')
                if descriptors['molecular_weight'] > 500:
                    warnings.append('High molecular weight (>500 Da)')
                if descriptors['tpsa'] > 90:
                    warnings.append('High TPSA (>90 A^2) - may have poor CNS penetration')
                if descriptors['logp'] < -1 or descriptors['logp'] > 5:
                    warnings.append(f'LogP outside typical range (-1 to 5): {descriptors["logp"]:.2f}')
                if descriptors['has_stereocenters'] and not self._is_defined_stereochemistry(smiles):
                    warnings.append('Undefined stereochemistry - predictions may vary by enantiomer')

                response['warnings'] = warnings

        return response

    def _is_defined_stereochemistry(self, smiles: str) -> bool:
        """Check if all stereocenters have defined configuration."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for _, stereo in chiral_centers:
            if stereo == '?':
                return False
        return True

    def predict_batch(self, smiles_list: List[str], return_details: bool = True) -> List[Dict]:
        """
        Predict transporter activity for multiple molecules.

        Args:
            smiles_list: List of SMILES strings
            return_details: If True, return detailed analysis for each molecule

        Returns:
            List of prediction results
        """
        results = []
        for smiles in smiles_list:
            result = self.predict(smiles, return_details=return_details)
            results.append(result)
        return results

    def compare_enantiomers(self, smiles1: str, smiles2: str) -> Dict:
        """
        Compare predictions for a pair of stereoisomers.

        Args:
            smiles1: First stereoisomer SMILES
            smiles2: Second stereoisomer SMILES

        Returns:
            Dict with comparison results for each target
        """
        pred1 = self.predict(smiles1, return_details=False)
        pred2 = self.predict(smiles2, return_details=False)

        if not pred1['success'] or not pred2['success']:
            return {
                'success': False,
                'error': 'One or both SMILES invalid',
                'smiles1': smiles1,
                'smiles2': smiles2,
            }

        comparisons = {}
        for target in self.targets:
            target_lower = target.lower()
            p1 = pred1[f'{target_lower}_score']
            p2 = pred2[f'{target_lower}_score']

            comparisons[target] = {
                'prob1': p1,
                'prob2': p2,
                'ratio': p1 / (p2 + 1e-8),
                'difference': p1 - p2,
                'more_active': smiles1 if p1 > p2 else smiles2,
            }

        return {
            'success': True,
            'smiles1': smiles1,
            'smiles2': smiles2,
            'comparisons': comparisons,
        }


def format_prediction_output(result: Dict) -> None:
    """Pretty print prediction results."""
    if not result['success']:
        print(f"FAILED: {result.get('error', 'Unknown error')}")
        return

    print(f"\nSMILES: {result['smiles']}")
    print(f"Category: {result['category']}")
    print(f"\nTransporter Predictions:")

    for target in ['DAT', 'NET', 'SERT']:
        target_lower = target.lower()
        score = result[f'{target_lower}_score']
        pred = result[f'{target_lower}_prediction']
        probs = result[f'{target_lower}_probs']

        # Format prediction line
        pred_symbol = {'inactive': '-', 'blocker': 'B', 'substrate': 'S'}[pred]
        print(f"  {target}: [{pred_symbol}] {pred.upper()} (substrate prob: {score:.3f})")
        print(f"       Probabilities: inactive={probs['inactive_prob']:.3f}, "
              f"blocker={probs['blocker_prob']:.3f}, substrate={probs['substrate_prob']:.3f}")

    if 'molecular_descriptors' in result:
        desc = result['molecular_descriptors']
        print(f"\nMolecular Properties:")
        print(f"  Molecular Weight: {desc['molecular_weight']:.1f} Da")
        print(f"  LogP: {desc['logp']:.2f}")
        print(f"  TPSA: {desc['tpsa']:.1f} A^2")
        print(f"  H-bond Donors: {desc['num_h_donors']}")
        print(f"  H-bond Acceptors: {desc['num_h_acceptors']}")
        print(f"  Basic Nitrogen: {desc['has_basic_nitrogen']}")
        print(f"  Stereocenters: {desc['num_stereocenters']}")
        print(f"  CNS Drug-like: {desc['cns_drug_like']}")

        if desc.get('is_amphetamine_like'):
            print(f"  Structure Type: Amphetamine-like")
        elif desc.get('is_phenethylamine'):
            print(f"  Structure Type: Phenethylamine")
        elif desc.get('is_cathinone_like'):
            print(f"  Structure Type: Cathinone-like")

    if result.get('warnings'):
        print(f"\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")

    print("-" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("StereoGNN Transporter Predictor - Testing")
    print("=" * 70)

    # Initialize predictor
    predictor = TransporterGNNPredictor()

    # Test compounds with known activities
    test_compounds = [
        ('C[C@H](N)Cc1ccccc1', 'd-Amphetamine', 'DAT/NET substrate'),
        ('C[C@@H](N)Cc1ccccc1', 'l-Amphetamine', 'Less active enantiomer'),
        ('C[C@H](NC)Cc1ccccc1', 'd-Methamphetamine', 'DAT/NET substrate'),
        ('NCCc1ccc(O)c(O)c1', 'Dopamine', 'Natural DAT substrate'),
        ('NC[C@H](O)c1ccc(O)c(O)c1', 'Norepinephrine', 'Natural NET substrate'),
        ('NCCc1c[nH]c2ccc(O)cc12', 'Serotonin', 'Natural SERT substrate'),
        ('COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3', 'Cocaine', 'Blocker (not substrate)'),
        ('Cn1cnc2c1c(=O)n(C)c(=O)n2C', 'Caffeine', 'Inactive'),
    ]

    print(f"\nTesting {len(test_compounds)} compounds:")
    print("=" * 70)

    for smiles, name, expected in test_compounds:
        print(f"\n{name} (Expected: {expected}):")
        result = predictor.predict(smiles, return_details=True)
        format_prediction_output(result)

    # Test enantiomer comparison
    print("\n" + "=" * 70)
    print("ENANTIOMER COMPARISON:")
    print("=" * 70)

    comparison = predictor.compare_enantiomers(
        "C[C@H](N)Cc1ccccc1",   # d-Amphetamine
        "C[C@@H](N)Cc1ccccc1",  # l-Amphetamine
    )

    if comparison['success']:
        print(f"\nd-Amphetamine vs l-Amphetamine:")
        for target, data in comparison['comparisons'].items():
            print(f"  {target}: d={data['prob1']:.3f}, l={data['prob2']:.3f}, ratio={data['ratio']:.2f}x")

    # Batch prediction test
    print("\n" + "=" * 70)
    print("BATCH PREDICTION:")
    print("=" * 70)

    batch_smiles = [s for s, _, _ in test_compounds[:4]]
    batch_results = predictor.predict_batch(batch_smiles, return_details=False)

    print(f"\nBatch results:")
    for i, (result, (_, name, _)) in enumerate(zip(batch_results, test_compounds[:4])):
        if result['success']:
            print(f"{i+1}. {name}: {result['category']}")
            print(f"   DAT: {result['dat_score']:.3f}, NET: {result['net_score']:.3f}, SERT: {result['sert_score']:.3f}")

    print("\n" + "=" * 70)
    print("Transporter GNN Predictor Ready!")
    print("=" * 70)
