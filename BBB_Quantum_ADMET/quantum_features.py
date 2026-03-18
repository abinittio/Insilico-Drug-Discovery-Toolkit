"""
Quantum-Enhanced Molecular Feature Extraction (34 dimensions)

Uses RDKit ETKDG for 3D conformer generation and computes:
- 15 atomic features
- 13 quantum descriptors (HOMO/LUMO approximations, Fukui, etc.)
- 6 stereochemistry features

This is INDEPENDENT from BBB_System.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from typing import Optional, List, Dict, Tuple
import warnings

from config import (
    ELECTRONEGATIVITY, POLARIZABILITY,
    IONIZATION_ENERGY, ELECTRON_AFFINITY,
    QUANTUM_CONFIG
)


class QuantumFeatureExtractor:
    """
    Extract 34-dimensional quantum-enhanced features for molecular graphs.

    Features:
        1-15:  Atomic properties
        16-28: Quantum descriptors (ETKDG-based approximations)
        29-34: Stereochemistry
    """

    def __init__(self, use_etkdg: bool = True, random_seed: int = 42):
        self.use_etkdg = use_etkdg
        self.random_seed = random_seed

    def generate_conformer(self, mol) -> bool:
        """Generate 3D conformer using ETKDG."""
        if mol is None:
            return False

        try:
            # Add hydrogens for accurate 3D
            mol = Chem.AddHs(mol)

            # ETKDG parameters
            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.maxIterations = QUANTUM_CONFIG.max_iterations
            params.pruneRmsThresh = QUANTUM_CONFIG.prune_rms_threshold

            # Generate conformer
            result = AllChem.EmbedMolecule(mol, params)

            if result == -1:
                # Fallback to random coordinates
                AllChem.EmbedMolecule(mol, randomSeed=self.random_seed)

            # Optimize geometry
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass  # Continue even if optimization fails

            return True

        except Exception as e:
            warnings.warn(f"Conformer generation failed: {e}")
            return False

    def compute_gasteiger_charges(self, mol) -> np.ndarray:
        """Compute Gasteiger partial charges."""
        try:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                charge = float(atom.GetProp('_GasteigerCharge'))
                if np.isnan(charge) or np.isinf(charge):
                    charge = 0.0
                charges.append(charge)
            return np.array(charges)
        except:
            return np.zeros(mol.GetNumAtoms())

    def compute_fukui_indices(self, mol, charges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Approximate Fukui indices from partial charges.

        f+ (nucleophilic attack): Where electrons go
        f- (electrophilic attack): Where electrons come from
        f0 (radical attack): Average
        """
        n_atoms = mol.GetNumAtoms()

        # Approximate using charge differences and electronegativity
        fukui_plus = np.zeros(n_atoms)   # Susceptibility to nucleophilic attack
        fukui_minus = np.zeros(n_atoms)  # Susceptibility to electrophilic attack
        fukui_zero = np.zeros(n_atoms)   # Radical susceptibility

        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            en = ELECTRONEGATIVITY.get(atomic_num, 2.5)
            charge = charges[i]

            # Higher electronegativity + negative charge = better nucleophile
            fukui_minus[i] = max(0, (en - 2.5) / 2.0 - charge * 0.5)

            # Lower electronegativity + positive charge = better electrophile
            fukui_plus[i] = max(0, (2.5 - en) / 2.0 + charge * 0.5)

            # Radical is average
            fukui_zero[i] = (fukui_plus[i] + fukui_minus[i]) / 2.0

        # Normalize
        if fukui_plus.sum() > 0:
            fukui_plus /= fukui_plus.sum()
        if fukui_minus.sum() > 0:
            fukui_minus /= fukui_minus.sum()
        if fukui_zero.sum() > 0:
            fukui_zero /= fukui_zero.sum()

        return fukui_plus, fukui_minus, fukui_zero

    def compute_orbital_energies(self, mol) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate HOMO/LUMO energies per atom using ionization energies
        and electron affinities.
        """
        n_atoms = mol.GetNumAtoms()
        homo_approx = np.zeros(n_atoms)
        lumo_approx = np.zeros(n_atoms)

        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()

            # HOMO approximation from ionization energy (negative = bound)
            ie = IONIZATION_ENERGY.get(atomic_num, 10.0)
            homo_approx[i] = -ie / 15.0  # Normalize to roughly [-1, 0]

            # LUMO approximation from electron affinity
            ea = ELECTRON_AFFINITY.get(atomic_num, 1.0)
            lumo_approx[i] = -ea / 5.0  # Normalize to roughly [-1, 0]

            # Adjust based on hybridization and charge
            hybridization = atom.GetHybridization()
            charge = atom.GetFormalCharge()

            if hybridization == Chem.HybridizationType.SP:
                homo_approx[i] -= 0.1  # More stable
                lumo_approx[i] -= 0.1
            elif hybridization == Chem.HybridizationType.SP2:
                homo_approx[i] -= 0.05
                lumo_approx[i] -= 0.05

            # Charge effects
            homo_approx[i] -= charge * 0.1
            lumo_approx[i] -= charge * 0.1

        return homo_approx, lumo_approx

    def compute_softness_hardness(self, homo: np.ndarray, lumo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute chemical softness and hardness.

        Hardness eta = (LUMO - HOMO) / 2
        Softness S = 1 / (2 * eta)
        """
        gap = lumo - homo
        gap = np.clip(gap, 0.01, None)  # Avoid division by zero

        hardness = gap / 2.0
        softness = 1.0 / (2.0 * hardness)

        # Normalize
        softness = np.clip(softness, 0, 10) / 10.0

        return softness, hardness

    def compute_electrophilicity(self, homo: np.ndarray, lumo: np.ndarray) -> np.ndarray:
        """
        Compute electrophilicity index.

        omega = mu^2 / (2 * eta)
        where mu = (HOMO + LUMO) / 2 (chemical potential)
        """
        mu = (homo + lumo) / 2.0  # Chemical potential
        gap = lumo - homo
        gap = np.clip(gap, 0.01, None)
        eta = gap / 2.0  # Hardness

        electrophilicity = (mu ** 2) / (2.0 * eta)

        # Normalize
        return np.clip(electrophilicity, 0, 5) / 5.0

    def get_atomic_features(self, atom) -> List[float]:
        """Extract 15 atomic features."""
        features = []

        # 1. Atomic number (normalized)
        features.append(atom.GetAtomicNum() / 100.0)

        # 2. Degree
        features.append(atom.GetDegree() / 6.0)

        # 3. Formal charge
        features.append(atom.GetFormalCharge() / 2.0)

        # 4. Hybridization
        hybridization_map = {
            Chem.HybridizationType.S: 0,
            Chem.HybridizationType.SP: 1,
            Chem.HybridizationType.SP2: 2,
            Chem.HybridizationType.SP3: 3,
            Chem.HybridizationType.SP3D: 4,
            Chem.HybridizationType.SP3D2: 5,
        }
        features.append(hybridization_map.get(atom.GetHybridization(), 0) / 5.0)

        # 5. Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)

        # 6. Ring membership
        features.append(1.0 if atom.IsInRing() else 0.0)

        # 7. Implicit H count
        features.append(atom.GetTotalNumHs() / 4.0)

        # 8. Total valence
        features.append(atom.GetTotalValence() / 6.0)

        # 9. Atomic mass (normalized)
        features.append(atom.GetMass() / 200.0)

        # 10. Electronegativity
        atomic_num = atom.GetAtomicNum()
        en = ELECTRONEGATIVITY.get(atomic_num, 2.5)
        features.append(en / 4.0)

        # 11. Polarizability
        pol = POLARIZABILITY.get(atomic_num, 1.5)
        features.append(pol / 6.0)

        # 12. H-bond donor
        is_donor = 0.0
        if atomic_num in [7, 8] and atom.GetTotalNumHs() > 0:
            is_donor = 1.0
        features.append(is_donor)

        # 13. H-bond acceptor
        is_acceptor = 0.0
        if atomic_num == 7 and atom.GetDegree() < 4 and atom.GetFormalCharge() <= 0:
            is_acceptor = 1.0
        elif atomic_num == 8 and atom.GetFormalCharge() <= 0:
            is_acceptor = 1.0
        features.append(is_acceptor)

        # 14. Is polar atom (N, O, S, P)
        features.append(1.0 if atomic_num in [7, 8, 15, 16] else 0.0)

        # 15. Ring size (0 if not in ring)
        ring_size = 0
        if atom.IsInRing():
            for size in [3, 4, 5, 6, 7, 8]:
                if atom.IsInRingSize(size):
                    ring_size = size
                    break
        features.append(ring_size / 8.0)

        return features

    def get_stereo_features(self, atom, mol) -> List[float]:
        """Extract 6 stereochemistry features."""
        features = [0.0] * 6

        # Chiral center info
        chiral_tag = atom.GetChiralTag()

        # 1. Is chiral center
        if chiral_tag != Chem.ChiralType.CHI_UNSPECIFIED:
            features[0] = 1.0

        # 2. R configuration
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            features[1] = 1.0

        # 3. S configuration
        if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            features[2] = 1.0

        # E/Z for atoms in double bonds
        atom_idx = atom.GetIdx()
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = bond.GetStereo()

                # 4. Part of E/Z bond
                if stereo in [Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ]:
                    features[3] = 1.0

                # 5. E configuration
                if stereo == Chem.BondStereo.STEREOE:
                    features[4] = 1.0

                # 6. Z configuration
                if stereo == Chem.BondStereo.STEREOZ:
                    features[5] = 1.0

        return features

    def mol_to_graph(self, smiles: str, targets: Optional[Dict[str, float]] = None) -> Optional[Data]:
        """
        Convert SMILES to PyTorch Geometric Data object with 34 features.

        Args:
            smiles: SMILES string
            targets: Dict of ADMET endpoint values (for training)

        Returns:
            Data object or None if conversion fails
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate 3D conformer
        if self.use_etkdg:
            mol_3d = Chem.AddHs(mol)
            self.generate_conformer(mol_3d)
        else:
            mol_3d = mol

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None

        # Compute Gasteiger charges
        charges = self.compute_gasteiger_charges(mol)

        # Compute Fukui indices
        fukui_plus, fukui_minus, fukui_zero = self.compute_fukui_indices(mol, charges)

        # Compute orbital approximations
        homo, lumo = self.compute_orbital_energies(mol)

        # Compute softness/hardness
        softness, hardness = self.compute_softness_hardness(homo, lumo)

        # Compute electrophilicity
        electrophilicity = self.compute_electrophilicity(homo, lumo)

        # Build node features
        node_features = []

        for i, atom in enumerate(mol.GetAtoms()):
            # Atomic features (1-15)
            atomic = self.get_atomic_features(atom)

            # Quantum features (16-28) - 13 features
            quantum = [
                charges[i],                    # 16. Gasteiger charge
                homo[i],                       # 17. HOMO approximation
                lumo[i],                       # 18. LUMO approximation
                lumo[i] - homo[i],             # 19. HOMO-LUMO gap
                fukui_plus[i],                 # 20. Fukui f+
                fukui_minus[i],                # 21. Fukui f-
                fukui_zero[i],                 # 22. Fukui f0
                softness[i],                   # 23. Chemical softness
                hardness[i],                   # 24. Chemical hardness
                electrophilicity[i],           # 25. Electrophilicity index
                (homo[i] + lumo[i]) / 2.0,     # 26. Chemical potential (mu)
                charges[i] * softness[i],      # 27. Local softness
                abs(charges[i]) * fukui_zero[i],  # 28. Reactivity index
            ]

            # Stereo features (29-34)
            stereo = self.get_stereo_features(atom, mol)

            # Combine all features
            node_features.append(atomic + quantum + stereo)

        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)

        # Build edge index (bonds)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected

        if len(edge_index) == 0:
            # Single atom molecule
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create Data object
        data = Data(x=x, edge_index=edge_index)

        # Add SMILES for reference
        data.smiles = smiles

        # Add targets if provided
        if targets is not None:
            for endpoint, value in targets.items():
                if value is not None and not np.isnan(value):
                    setattr(data, endpoint, torch.tensor([value], dtype=torch.float))

        # Add molecular descriptors for self-supervised targets
        data.mol_weight = torch.tensor([Descriptors.MolWt(mol) / 500.0], dtype=torch.float)
        data.num_atoms = torch.tensor([n_atoms / 50.0], dtype=torch.float)
        data.logp = torch.tensor([Descriptors.MolLogP(mol) / 5.0], dtype=torch.float)

        return data


def batch_smiles_to_graphs(
    smiles_list: List[str],
    targets_list: Optional[List[Dict[str, float]]] = None,
    use_etkdg: bool = True,
    verbose: bool = True
) -> List[Data]:
    """
    Convert list of SMILES to graphs.

    Args:
        smiles_list: List of SMILES strings
        targets_list: Optional list of target dicts (one per SMILES)
        use_etkdg: Whether to generate 3D conformers
        verbose: Print progress

    Returns:
        List of Data objects (None entries filtered out)
    """
    extractor = QuantumFeatureExtractor(use_etkdg=use_etkdg)
    graphs = []

    for i, smiles in enumerate(smiles_list):
        targets = targets_list[i] if targets_list else None
        graph = extractor.mol_to_graph(smiles, targets)

        if graph is not None:
            graphs.append(graph)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(smiles_list)} ({len(graphs)} valid)")

    if verbose:
        print(f"Converted {len(graphs)}/{len(smiles_list)} molecules to graphs")

    return graphs


# Test
if __name__ == "__main__":
    print("Testing Quantum Feature Extractor...")
    print("=" * 60)

    extractor = QuantumFeatureExtractor(use_etkdg=True)

    test_molecules = [
        ("CCO", "Ethanol"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("C[C@H](Cc1ccccc1)N", "Amphetamine (R)"),
    ]

    for smiles, name in test_molecules:
        graph = extractor.mol_to_graph(smiles)

        if graph is not None:
            print(f"\n{name} ({smiles})")
            print(f"  Nodes: {graph.x.shape[0]}")
            print(f"  Features per node: {graph.x.shape[1]}")
            print(f"  Edges: {graph.edge_index.shape[1]}")
            print(f"  Feature range: [{graph.x.min():.3f}, {graph.x.max():.3f}]")
        else:
            print(f"\n{name}: Failed to convert")

    print("\n" + "=" * 60)
    print("Feature extraction test complete!")
