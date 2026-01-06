"""
Molecular Featurization for StereoGNN
=====================================

This module converts molecules into graph representations suitable for GNN processing.
CRITICAL: Explicit stereochemistry encoding is the key innovation.

Features encoded:
1. Atom features (element, hybridization, aromaticity, etc.)
2. Bond features (bond type, conjugation, ring membership, etc.)
3. STEREOCHEMISTRY features:
   - Tetrahedral chirality (R/S, CW/CCW)
   - Double bond geometry (E/Z, cis/trans)
   - Ring conformations
   - 3D coordinates from conformer generation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import ChiralType, BondStereo, BondType, HybridizationType
from rdkit.Chem.rdchem import StereoGroup, StereoGroupType

from config import CONFIG, StereoConfig


# Atom feature dimensions
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # All elements
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.UNSPECIFIED,
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

# Chiral tag encoding
CHIRAL_TAGS = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: 1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    ChiralType.CHI_OTHER: 3,
    ChiralType.CHI_TETRAHEDRAL: 4,
    ChiralType.CHI_ALLENE: 5,
    ChiralType.CHI_SQUAREPLANAR: 6,
    ChiralType.CHI_TRIGONALBIPYRAMIDAL: 7,
    ChiralType.CHI_OCTAHEDRAL: 8,
}

# Bond stereo encoding
BOND_STEREO = {
    BondStereo.STEREONONE: 0,
    BondStereo.STEREOANY: 1,
    BondStereo.STEREOZ: 2,
    BondStereo.STEREOE: 3,
    BondStereo.STEREOCIS: 4,
    BondStereo.STEREOTRANS: 5,
}

# Bond type encoding
BOND_TYPES = {
    BondType.SINGLE: 0,
    BondType.DOUBLE: 1,
    BondType.TRIPLE: 2,
    BondType.AROMATIC: 3,
}


def one_hot_encoding(value, allowable_set: List) -> List[int]:
    """One-hot encode a value from an allowable set."""
    encoding = [0] * (len(allowable_set) + 1)  # +1 for unknown
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    else:
        encoding[-1] = 1  # Unknown
    return encoding


@dataclass
class StereoFeatures:
    """Container for stereochemistry-specific features."""
    # Per-atom stereo features
    chiral_tag: np.ndarray          # One-hot encoded chiral tag
    rs_config: np.ndarray           # R=1, S=-1, undefined=0
    is_stereocenter: np.ndarray     # Binary

    # Per-bond stereo features
    bond_stereo: np.ndarray         # One-hot encoded E/Z
    is_stereo_bond: np.ndarray      # Binary

    # Global stereo features
    num_stereocenters: int
    num_stereo_bonds: int
    has_undefined_stereo: bool


class StereoFeaturizer:
    """
    Extracts stereochemistry features from molecules.

    This is the KEY INNOVATION of our approach.
    """

    def __init__(self, config: StereoConfig = None):
        self.config = config or CONFIG.stereo

    def featurize(self, mol: Chem.Mol) -> StereoFeatures:
        """Extract all stereochemistry features from a molecule."""
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        # Initialize arrays
        chiral_tags = np.zeros((n_atoms, len(CHIRAL_TAGS)))
        rs_configs = np.zeros(n_atoms)
        is_stereocenter = np.zeros(n_atoms)

        bond_stereo = np.zeros((n_bonds, len(BOND_STEREO)))
        is_stereo_bond = np.zeros(n_bonds)

        # Get CIP codes if available
        try:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        except Exception:
            pass

        # Find chiral centers
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        chiral_center_dict = {idx: code for idx, code in chiral_centers}

        has_undefined = False

        # Process atoms
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()

            # Chiral tag (one-hot)
            chiral_tag = atom.GetChiralTag()
            tag_idx = CHIRAL_TAGS.get(chiral_tag, 0)
            chiral_tags[idx, tag_idx] = 1

            # R/S configuration
            if idx in chiral_center_dict:
                is_stereocenter[idx] = 1
                code = chiral_center_dict[idx]
                if code == 'R':
                    rs_configs[idx] = 1
                elif code == 'S':
                    rs_configs[idx] = -1
                elif code == '?':
                    has_undefined = True
                    rs_configs[idx] = 0

        # Process bonds
        for bond in mol.GetBonds():
            idx = bond.GetIdx()

            stereo = bond.GetStereo()
            stereo_idx = BOND_STEREO.get(stereo, 0)
            bond_stereo[idx, stereo_idx] = 1

            if stereo != BondStereo.STEREONONE:
                is_stereo_bond[idx] = 1

        return StereoFeatures(
            chiral_tag=chiral_tags,
            rs_config=rs_configs,
            is_stereocenter=is_stereocenter,
            bond_stereo=bond_stereo,
            is_stereo_bond=is_stereo_bond,
            num_stereocenters=int(is_stereocenter.sum()),
            num_stereo_bonds=int(is_stereo_bond.sum()),
            has_undefined_stereo=has_undefined,
        )


class AtomFeaturizer:
    """Featurizes atoms for GNN input."""

    def __init__(self):
        self.stereo_featurizer = StereoFeaturizer()

    def featurize_atom(self, atom: Chem.Atom) -> np.ndarray:
        """Generate feature vector for a single atom."""
        features = []

        # Basic features
        features.extend(one_hot_encoding(
            atom.GetAtomicNum(),
            ATOM_FEATURES['atomic_num'][:36]  # Up to Kr for efficiency
        ))
        features.extend(one_hot_encoding(
            atom.GetTotalDegree(),
            ATOM_FEATURES['degree']
        ))
        features.extend(one_hot_encoding(
            atom.GetFormalCharge(),
            ATOM_FEATURES['formal_charge']
        ))
        features.extend(one_hot_encoding(
            atom.GetTotalNumHs(),
            ATOM_FEATURES['num_hs']
        ))
        features.extend(one_hot_encoding(
            atom.GetHybridization(),
            ATOM_FEATURES['hybridization']
        ))

        # Binary features
        features.append(int(atom.GetIsAromatic()))
        features.append(int(atom.IsInRing()))
        features.append(int(atom.IsInRingSize(3)))
        features.append(int(atom.IsInRingSize(4)))
        features.append(int(atom.IsInRingSize(5)))
        features.append(int(atom.IsInRingSize(6)))

        # Electronegativity proxy (group-based)
        atomic_num = atom.GetAtomicNum()
        features.append(self._get_electronegativity(atomic_num))

        # Mass
        features.append(atom.GetMass() / 100.0)  # Normalize

        return np.array(features, dtype=np.float32)

    def _get_electronegativity(self, atomic_num: int) -> float:
        """Get approximate electronegativity."""
        # Pauling electronegativity (normalized)
        en_dict = {
            1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
            15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
        }
        return en_dict.get(atomic_num, 2.0) / 4.0

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        """Generate feature matrix for all atoms in molecule."""
        features = []
        for atom in mol.GetAtoms():
            features.append(self.featurize_atom(atom))
        # Returns 75 features to match config.py atom_feature_dim
        return np.stack(features) if features else np.zeros((0, 75))


class BondFeaturizer:
    """Featurizes bonds for GNN input."""

    def featurize_bond(self, bond: Chem.Bond) -> np.ndarray:
        """Generate feature vector for a single bond."""
        features = []

        # Bond type
        features.extend(one_hot_encoding(
            bond.GetBondType(),
            list(BOND_TYPES.keys())
        ))

        # Binary features
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))
        features.append(int(bond.IsInRingSize(3)))
        features.append(int(bond.IsInRingSize(4)))
        features.append(int(bond.IsInRingSize(5)))
        features.append(int(bond.IsInRingSize(6)))

        return np.array(features, dtype=np.float32)

    def featurize_mol(self, mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate edge features for molecule.

        Returns:
            edge_index: [2, num_edges] tensor of edge indices
            edge_attr: [num_edges, num_features] tensor of edge features
        """
        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_features = self.featurize_bond(bond)

            # Add both directions (undirected graph)
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)

        if edge_indices:
            edge_index = np.array(edge_indices).T
            edge_attr = np.stack(edge_attrs)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 18), dtype=np.float32)  # 11 base + 7 stereo

        return edge_index, edge_attr


class ConformerGenerator:
    """Generates 3D conformers for molecules."""

    def __init__(self, num_conformers: int = 5, seed: int = 42):
        self.num_conformers = num_conformers
        self.seed = seed

    def generate(self, mol: Chem.Mol) -> Optional[np.ndarray]:
        """
        Generate 3D coordinates for molecule.

        Returns:
            coords: [num_atoms, 3] array of 3D coordinates
        """
        mol = Chem.AddHs(mol)

        try:
            # Generate conformers
            params = AllChem.ETKDGv3()
            params.randomSeed = self.seed
            params.numThreads = 0  # Use all available threads

            conf_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.num_conformers,
                params=params,
            )

            if not conf_ids:
                # Fallback to 2D -> 3D
                AllChem.EmbedMolecule(mol, params)

            # Optimize
            AllChem.MMFFOptimizeMoleculeConfs(mol)

            # Get coordinates of first conformer
            conf = mol.GetConformer(0)
            coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])

            # Remove Hs, keep only heavy atoms
            mol_no_h = Chem.RemoveHs(mol)
            n_heavy = mol_no_h.GetNumHeavyAtoms()

            return coords[:n_heavy]

        except Exception as e:
            return None


class MoleculeGraphFeaturizer:
    """
    Complete molecular graph featurizer for StereoGNN.

    Produces PyTorch Geometric Data objects with:
    - Node features (atoms + stereochemistry)
    - Edge features (bonds + stereo bonds)
    - 3D coordinates (optional)
    - Global molecular features
    """

    def __init__(self, config: StereoConfig = None, use_3d: bool = True):
        self.config = config or CONFIG.stereo
        self.atom_featurizer = AtomFeaturizer()
        self.bond_featurizer = BondFeaturizer()
        self.stereo_featurizer = StereoFeaturizer()
        self.conformer_gen = ConformerGenerator() if use_3d else None
        self.use_3d = use_3d

    def featurize(
        self,
        smiles: str,
        labels: Optional[Dict[str, int]] = None,
    ) -> Optional[Data]:
        """
        Convert SMILES to PyTorch Geometric Data object.

        Args:
            smiles: SMILES string
            labels: Optional dict mapping target -> label (0, 1, or 2)

        Returns:
            Data object or None if featurization fails
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        atom_features = self.atom_featurizer.featurize_mol(mol)

        # Get stereo features
        stereo_features = self.stereo_featurizer.featurize(mol)

        # Concatenate atom and stereo features
        node_features = np.concatenate([
            atom_features,
            stereo_features.chiral_tag,  # [N, 9]
            stereo_features.rs_config.reshape(-1, 1),  # [N, 1]
            stereo_features.is_stereocenter.reshape(-1, 1),  # [N, 1]
        ], axis=1)

        # Get bond features
        edge_index, edge_attr = self.bond_featurizer.featurize_mol(mol)

        # Add stereo bond features
        # Need to expand stereo features to match edge_attr shape
        n_bonds = mol.GetNumBonds()
        if n_bonds > 0:
            stereo_bond_attr = []
            for bond in mol.GetBonds():
                bond_stereo_feat = stereo_features.bond_stereo[bond.GetIdx()]
                is_stereo = stereo_features.is_stereo_bond[bond.GetIdx()]
                combined = np.concatenate([bond_stereo_feat, [is_stereo]])
                # Add for both directions
                stereo_bond_attr.append(combined)
                stereo_bond_attr.append(combined)

            stereo_bond_attr = np.stack(stereo_bond_attr)
            edge_attr = np.concatenate([edge_attr, stereo_bond_attr], axis=1)

        # Get 3D coordinates if requested
        pos = None
        if self.use_3d and self.conformer_gen:
            pos = self.conformer_gen.generate(mol)
            if pos is not None:
                pos = torch.tensor(pos, dtype=torch.float32)

        # Create Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )

        if pos is not None:
            data.pos = pos

        # Add stereochemistry metadata
        data.num_stereocenters = stereo_features.num_stereocenters
        data.num_stereo_bonds = stereo_features.num_stereo_bonds
        data.has_undefined_stereo = stereo_features.has_undefined_stereo

        # Add labels if provided
        if labels:
            data.y_dat = torch.tensor([labels.get('DAT', -1)], dtype=torch.long)
            data.y_net = torch.tensor([labels.get('NET', -1)], dtype=torch.long)
            data.y_sert = torch.tensor([labels.get('SERT', -1)], dtype=torch.long)

        # Add SMILES for reference
        data.smiles = smiles

        return data

    def featurize_batch(
        self,
        smiles_list: List[str],
        labels_list: Optional[List[Dict[str, int]]] = None,
    ) -> List[Data]:
        """Featurize a batch of molecules."""
        data_list = []

        for i, smi in enumerate(smiles_list):
            labels = labels_list[i] if labels_list else None
            data = self.featurize(smi, labels)
            if data is not None:
                data_list.append(data)

        return data_list


def get_feature_dimensions() -> Dict[str, int]:
    """Get the dimensions of all feature types."""
    # Create a test molecule
    test_mol = Chem.MolFromSmiles("CC(N)Cc1ccccc1")  # Amphetamine

    featurizer = MoleculeGraphFeaturizer(use_3d=False)
    data = featurizer.featurize("CC(N)Cc1ccccc1")

    return {
        'node_features': data.x.shape[1],
        'edge_features': data.edge_attr.shape[1],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("StereoGNN Featurizer Test")
    print("=" * 60)

    # Test molecules with different stereo features
    test_mols = [
        ("C[C@H](N)Cc1ccccc1", "d-Amphetamine"),
        ("C[C@@H](N)Cc1ccccc1", "l-Amphetamine"),
        ("CC(N)Cc1ccccc1", "Racemic Amphetamine"),
        ("C/C=C/C", "(E)-2-butene"),
        ("C/C=C\\C", "(Z)-2-butene"),
    ]

    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    for smi, name in test_mols:
        data = featurizer.featurize(smi)
        if data:
            print(f"\n{name} ({smi}):")
            print(f"  Nodes: {data.x.shape}")
            print(f"  Edges: {data.edge_attr.shape}")
            print(f"  Stereocenters: {data.num_stereocenters}")
            print(f"  Stereo bonds: {data.num_stereo_bonds}")

    print("\n" + "=" * 60)
    print("Feature Dimensions:", get_feature_dimensions())
