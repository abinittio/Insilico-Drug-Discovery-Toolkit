"""
PyTorch Dataset for Transporter Substrate Data
===============================================

Handles loading, preprocessing, and batching of molecular data
for the StereoGNN model.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from featurizer import MoleculeGraphFeaturizer
from config import CONFIG


class TransporterDataset(Dataset):
    """
    Dataset for transporter substrate/blocker classification.

    Supports:
    - Multi-task labels (DAT, NET, SERT)
    - Pre-featurization for efficiency
    - Data augmentation (stereo enumeration)
    """

    def __init__(
        self,
        data_path: Path,
        split: str = 'train',
        featurizer: Optional[MoleculeGraphFeaturizer] = None,
        transform: Optional[Callable] = None,
        pre_featurize: bool = True,
        use_3d: bool = True,
    ):
        """
        Args:
            data_path: Path to data directory
            split: 'train', 'val', or 'test'
            featurizer: Molecular featurizer (creates default if None)
            transform: Optional transform to apply to data
            pre_featurize: Whether to featurize all molecules upfront
            use_3d: Whether to use 3D coordinates
        """
        super().__init__()

        self.split = split
        self.transform = transform
        self.use_3d = use_3d

        # Load data
        split_path = data_path / f"{split}.parquet"
        if split_path.exists():
            self.df = pd.read_parquet(split_path)
        else:
            # Create empty DataFrame with expected columns
            self.df = pd.DataFrame(columns=['smiles', 'target', 'label', 'confidence'])

        # Create featurizer
        self.featurizer = featurizer or MoleculeGraphFeaturizer(use_3d=use_3d)

        # Get unique molecules
        self.molecules = self.df['smiles'].unique().tolist()

        # Create label lookup
        self.labels = self._create_label_lookup()

        # Pre-featurize if requested
        self.pre_featurized = {}
        if pre_featurize and len(self.molecules) > 0:
            self._pre_featurize()

    def _create_label_lookup(self) -> Dict[str, Dict[str, int]]:
        """Create a lookup from SMILES to labels for each target."""
        labels = {}

        for smi in self.molecules:
            mol_df = self.df[self.df['smiles'] == smi]

            labels[smi] = {
                'DAT': -1,  # -1 = unknown/missing
                'NET': -1,
                'SERT': -1,
            }

            for target in ['DAT', 'NET', 'SERT']:
                target_df = mol_df[mol_df['target'] == target]
                if len(target_df) > 0:
                    labels[smi][target] = int(target_df['label'].iloc[0])

        return labels

    def _pre_featurize(self):
        """Pre-compute features for all molecules."""
        print(f"Pre-featurizing {len(self.molecules)} molecules for {self.split}...")

        for smi in tqdm(self.molecules):
            labels = self.labels[smi]
            data = self.featurizer.featurize(smi, labels)
            if data is not None:
                self.pre_featurized[smi] = data

        print(f"Successfully featurized {len(self.pre_featurized)}/{len(self.molecules)}")

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Data:
        smiles = self.molecules[idx]

        # Get pre-featurized data or compute on the fly
        if smiles in self.pre_featurized:
            data = self.pre_featurized[smiles]
        else:
            labels = self.labels[smiles]
            data = self.featurizer.featurize(smiles, labels)
            if data is None:
                # Return dummy data if featurization fails
                data = self._create_dummy_data()

        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)

        return data

    def _create_dummy_data(self) -> Data:
        """Create dummy data for failed molecules."""
        return Data(
            x=torch.zeros((1, 89)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 18)),
            y_dat=torch.tensor([-1]),
            y_net=torch.tensor([-1]),
            y_sert=torch.tensor([-1]),
            smiles="",
        )

    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Compute class weights for each target based on label distribution."""
        weights = {}

        for target in ['DAT', 'NET', 'SERT']:
            target_df = self.df[self.df['target'] == target]
            labels = target_df['label'].values

            # Count classes
            counts = np.bincount(labels[labels >= 0], minlength=3)

            # Inverse frequency weighting
            if counts.sum() > 0:
                w = counts.sum() / (3 * counts + 1e-6)
                w = w / w.mean()
            else:
                w = np.ones(3)

            weights[target] = torch.tensor(w, dtype=torch.float32)

        return weights

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'split': self.split,
            'num_molecules': len(self.molecules),
            'num_records': len(self.df),
        }

        for target in ['DAT', 'NET', 'SERT']:
            target_df = self.df[self.df['target'] == target]
            stats[f'{target}_total'] = len(target_df)

            for label, name in [(0, 'inactive'), (1, 'blocker'), (2, 'substrate')]:
                stats[f'{target}_{name}'] = len(target_df[target_df['label'] == label])

        return stats


class StereoAugmentation:
    """
    Data augmentation via stereoisomer enumeration.

    For molecules with defined stereochemistry, randomly flip
    stereocenters during training to improve robustness.
    """

    def __init__(self, flip_prob: float = 0.1):
        self.flip_prob = flip_prob

    def __call__(self, data: Data) -> Data:
        # Only augment during training and if molecule has stereocenters
        if not hasattr(data, 'num_stereocenters') or data.num_stereocenters == 0:
            return data

        # With probability flip_prob, flip the R/S encoding
        if torch.rand(1).item() < self.flip_prob:
            # Flip the R/S configuration feature (last stereo feature)
            # This is at position -2 in node features (before is_stereocenter)
            data.x[:, -2] = -data.x[:, -2]

        return data


def collate_fn(batch: List[Data]) -> Batch:
    """Custom collate function for DataLoader."""
    return Batch.from_data_list(batch)


# =============================================================================
# KINETIC EXTENSION - Dataset for mechanistic parameter prediction
# =============================================================================

class KineticTransporterDataset(TransporterDataset):
    """
    Extended dataset for kinetic parameter prediction.

    Supports:
    - All original labels (activity classification)
    - Kinetic parameters: pKi, pIC50
    - Interaction mode labels
    - Kinetic bias values
    - Confidence scores for weak labels

    Data format (parquet/csv):
        smiles: SMILES string
        target: DAT/NET/SERT
        label: Activity class (0=inactive, 1=blocker, 2=substrate)
        pKi: Binding affinity (-log10(Ki)), NaN if missing
        pIC50: Functional potency (-log10(IC50)), NaN if missing
        interaction_mode: 0=substrate, 1=competitive, 2=non-competitive, 3=partial, -1=unknown
        kinetic_bias: Uptake preference (0-1), NaN if missing
        confidence: Label confidence (0-1), 1.0 if not specified
    """

    # Interaction mode encoding (matches KineticHead)
    MODE_SUBSTRATE = 0
    MODE_COMPETITIVE = 1
    MODE_NONCOMPETITIVE = 2
    MODE_PARTIAL = 3

    def __init__(
        self,
        data_path: Path,
        split: str = 'train',
        featurizer: Optional[MoleculeGraphFeaturizer] = None,
        transform: Optional[Callable] = None,
        pre_featurize: bool = True,
        use_3d: bool = True,
        kinetic_columns: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            data_path: Path to data directory
            split: 'train', 'val', or 'test'
            featurizer: Molecular featurizer
            transform: Optional transform
            pre_featurize: Whether to pre-compute features
            use_3d: Whether to use 3D coordinates
            kinetic_columns: Mapping of kinetic column names
                Defaults: {'pKi': 'pKi', 'pIC50': 'pIC50',
                          'mode': 'interaction_mode', 'bias': 'kinetic_bias'}
        """
        # Set kinetic column names before calling super().__init__
        self.kinetic_columns = kinetic_columns or {
            'pKi': 'pKi',
            'pIC50': 'pIC50',
            'mode': 'interaction_mode',
            'bias': 'kinetic_bias',
            'confidence': 'confidence',
        }

        # Call parent init (which calls _create_label_lookup and _pre_featurize)
        super().__init__(
            data_path=data_path,
            split=split,
            featurizer=featurizer,
            transform=transform,
            pre_featurize=False,  # We'll do it ourselves with kinetic labels
            use_3d=use_3d,
        )

        # Create extended label lookup with kinetic parameters
        self.kinetic_labels = self._create_kinetic_label_lookup()

        # Pre-featurize with kinetic labels
        if pre_featurize and len(self.molecules) > 0:
            self._pre_featurize_kinetic()

    def _create_kinetic_label_lookup(self) -> Dict[str, Dict[str, Dict]]:
        """Create lookup from SMILES to kinetic labels for each target."""
        kinetic_labels = {}

        for smi in self.molecules:
            mol_df = self.df[self.df['smiles'] == smi]

            kinetic_labels[smi] = {}

            for target in ['DAT', 'NET', 'SERT']:
                target_df = mol_df[mol_df['target'] == target]

                # Default values (missing)
                kinetic_labels[smi][target] = {
                    'class': -1,  # Activity class
                    'pKi': float('nan'),
                    'pIC50': float('nan'),
                    'mode': -1,  # Interaction mode
                    'kinetic_bias': float('nan'),
                    'confidence': 1.0,
                }

                if len(target_df) > 0:
                    row = target_df.iloc[0]

                    # Activity class
                    if 'label' in row:
                        kinetic_labels[smi][target]['class'] = int(row['label'])

                    # pKi
                    pki_col = self.kinetic_columns['pKi']
                    if pki_col in row and pd.notna(row[pki_col]):
                        kinetic_labels[smi][target]['pKi'] = float(row[pki_col])

                    # pIC50
                    pic50_col = self.kinetic_columns['pIC50']
                    if pic50_col in row and pd.notna(row[pic50_col]):
                        kinetic_labels[smi][target]['pIC50'] = float(row[pic50_col])

                    # Interaction mode
                    mode_col = self.kinetic_columns['mode']
                    if mode_col in row and pd.notna(row[mode_col]):
                        kinetic_labels[smi][target]['mode'] = int(row[mode_col])

                    # Kinetic bias
                    bias_col = self.kinetic_columns['bias']
                    if bias_col in row and pd.notna(row[bias_col]):
                        kinetic_labels[smi][target]['kinetic_bias'] = float(row[bias_col])

                    # Confidence
                    conf_col = self.kinetic_columns['confidence']
                    if conf_col in row and pd.notna(row[conf_col]):
                        kinetic_labels[smi][target]['confidence'] = float(row[conf_col])

        return kinetic_labels

    def _pre_featurize_kinetic(self):
        """Pre-compute features with kinetic labels."""
        print(f"Pre-featurizing {len(self.molecules)} molecules with kinetic labels for {self.split}...")

        for smi in tqdm(self.molecules):
            labels = self.labels[smi]  # Original activity labels
            kinetic = self.kinetic_labels[smi]

            data = self.featurizer.featurize(smi, labels)
            if data is not None:
                # Add kinetic labels to data object
                for target in ['DAT', 'NET', 'SERT']:
                    target_lower = target.lower()
                    k = kinetic[target]

                    # Kinetic regression targets (use NaN for missing)
                    setattr(data, f'y_{target_lower}_pki', torch.tensor([k['pKi']], dtype=torch.float32))
                    setattr(data, f'y_{target_lower}_pic50', torch.tensor([k['pIC50']], dtype=torch.float32))
                    setattr(data, f'y_{target_lower}_bias', torch.tensor([k['kinetic_bias']], dtype=torch.float32))

                    # Kinetic classification target
                    setattr(data, f'y_{target_lower}_mode', torch.tensor([k['mode']], dtype=torch.long))

                    # Confidence
                    setattr(data, f'y_{target_lower}_confidence', torch.tensor([k['confidence']], dtype=torch.float32))

                self.pre_featurized[smi] = data

        print(f"Successfully featurized {len(self.pre_featurized)}/{len(self.molecules)}")

    def __getitem__(self, idx: int) -> Data:
        smiles = self.molecules[idx]

        if smiles in self.pre_featurized:
            data = self.pre_featurized[smiles]
        else:
            # Featurize on the fly
            labels = self.labels[smiles]
            kinetic = self.kinetic_labels[smiles]

            data = self.featurizer.featurize(smiles, labels)
            if data is None:
                data = self._create_dummy_kinetic_data()
            else:
                # Add kinetic labels
                for target in ['DAT', 'NET', 'SERT']:
                    target_lower = target.lower()
                    k = kinetic[target]

                    setattr(data, f'y_{target_lower}_pki', torch.tensor([k['pKi']], dtype=torch.float32))
                    setattr(data, f'y_{target_lower}_pic50', torch.tensor([k['pIC50']], dtype=torch.float32))
                    setattr(data, f'y_{target_lower}_bias', torch.tensor([k['kinetic_bias']], dtype=torch.float32))
                    setattr(data, f'y_{target_lower}_mode', torch.tensor([k['mode']], dtype=torch.long))
                    setattr(data, f'y_{target_lower}_confidence', torch.tensor([k['confidence']], dtype=torch.float32))

        if self.transform is not None:
            data = self.transform(data)

        return data

    def _create_dummy_kinetic_data(self) -> Data:
        """Create dummy data with kinetic labels for failed molecules."""
        data = self._create_dummy_data()

        # Add kinetic labels (all missing)
        for target in ['dat', 'net', 'sert']:
            setattr(data, f'y_{target}_pki', torch.tensor([float('nan')], dtype=torch.float32))
            setattr(data, f'y_{target}_pic50', torch.tensor([float('nan')], dtype=torch.float32))
            setattr(data, f'y_{target}_bias', torch.tensor([float('nan')], dtype=torch.float32))
            setattr(data, f'y_{target}_mode', torch.tensor([-1], dtype=torch.long))
            setattr(data, f'y_{target}_confidence', torch.tensor([0.0], dtype=torch.float32))

        return data

    def get_kinetic_statistics(self) -> Dict:
        """Get statistics about kinetic label availability."""
        stats = self.get_statistics()

        for target in ['DAT', 'NET', 'SERT']:
            target_df = self.df[self.df['target'] == target]

            # Count available kinetic labels
            pki_col = self.kinetic_columns['pKi']
            pic50_col = self.kinetic_columns['pIC50']
            mode_col = self.kinetic_columns['mode']
            bias_col = self.kinetic_columns['bias']

            if pki_col in target_df.columns:
                stats[f'{target}_pKi_available'] = target_df[pki_col].notna().sum()
            else:
                stats[f'{target}_pKi_available'] = 0

            if pic50_col in target_df.columns:
                stats[f'{target}_pIC50_available'] = target_df[pic50_col].notna().sum()
            else:
                stats[f'{target}_pIC50_available'] = 0

            if mode_col in target_df.columns:
                stats[f'{target}_mode_available'] = (target_df[mode_col] >= 0).sum()
                # Mode distribution
                for mode_val, mode_name in [(0, 'substrate'), (1, 'competitive'),
                                             (2, 'noncompetitive'), (3, 'partial')]:
                    stats[f'{target}_mode_{mode_name}'] = (target_df[mode_col] == mode_val).sum()
            else:
                stats[f'{target}_mode_available'] = 0

            if bias_col in target_df.columns:
                stats[f'{target}_bias_available'] = target_df[bias_col].notna().sum()
            else:
                stats[f'{target}_bias_available'] = 0

        return stats


def create_kinetic_dataloaders(
    data_path: Path = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_3d: bool = True,
    augment: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train, val, test dataloaders for kinetic model.

    Args:
        data_path: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_3d: Whether to use 3D coordinates
        augment: Whether to apply data augmentation (training only)

    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    data_path = data_path or CONFIG.data.data_dir

    # Create shared featurizer
    featurizer = MoleculeGraphFeaturizer(use_3d=use_3d)

    # Create datasets
    train_transform = StereoAugmentation(flip_prob=0.1) if augment else None

    datasets = {
        'train': KineticTransporterDataset(
            data_path, 'train', featurizer,
            transform=train_transform, use_3d=use_3d
        ),
        'val': KineticTransporterDataset(
            data_path, 'val', featurizer,
            transform=None, use_3d=use_3d
        ),
        'test': KineticTransporterDataset(
            data_path, 'test', featurizer,
            transform=None, use_3d=use_3d
        ),
    }

    # Create dataloaders
    dataloaders = {
        'train': PyGDataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        'val': PyGDataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        'test': PyGDataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders


def batch_to_kinetic_targets(batch: Batch) -> Dict[str, torch.Tensor]:
    """
    Extract kinetic targets from a batch for loss computation.

    Args:
        batch: PyTorch Geometric Batch object

    Returns:
        Dict with target tensors for KineticMultiTaskLoss
    """
    targets = {}

    for task in ['DAT', 'NET', 'SERT']:
        task_lower = task.lower()

        # Activity class
        if hasattr(batch, f'y_{task_lower}'):
            targets[f'{task}_class'] = getattr(batch, f'y_{task_lower}')
        elif hasattr(batch, f'y_{task}'):
            targets[f'{task}_class'] = getattr(batch, f'y_{task}')

        # Kinetic regression targets
        if hasattr(batch, f'y_{task_lower}_pki'):
            targets[f'{task}_pKi'] = getattr(batch, f'y_{task_lower}_pki').squeeze()

        if hasattr(batch, f'y_{task_lower}_pic50'):
            targets[f'{task}_pIC50'] = getattr(batch, f'y_{task_lower}_pic50').squeeze()

        if hasattr(batch, f'y_{task_lower}_bias'):
            targets[f'{task}_kinetic_bias'] = getattr(batch, f'y_{task_lower}_bias').squeeze()

        # Interaction mode
        if hasattr(batch, f'y_{task_lower}_mode'):
            targets[f'{task}_mode'] = getattr(batch, f'y_{task_lower}_mode').squeeze()

        # Confidence
        if hasattr(batch, f'y_{task_lower}_confidence'):
            targets[f'{task}_confidence'] = getattr(batch, f'y_{task_lower}_confidence').squeeze()

    return targets


def create_dataloaders(
    data_path: Path = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_3d: bool = True,
    augment: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train, val, test dataloaders.

    Args:
        data_path: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_3d: Whether to use 3D coordinates
        augment: Whether to apply data augmentation (training only)

    Returns:
        Dict with 'train', 'val', 'test' dataloaders
    """
    data_path = data_path or CONFIG.data.data_dir

    # Create shared featurizer
    featurizer = MoleculeGraphFeaturizer(use_3d=use_3d)

    # Create datasets
    train_transform = StereoAugmentation(flip_prob=0.1) if augment else None

    datasets = {
        'train': TransporterDataset(
            data_path, 'train', featurizer,
            transform=train_transform, use_3d=use_3d
        ),
        'val': TransporterDataset(
            data_path, 'val', featurizer,
            transform=None, use_3d=use_3d
        ),
        'test': TransporterDataset(
            data_path, 'test', featurizer,
            transform=None, use_3d=use_3d
        ),
    }

    # Create dataloaders
    dataloaders = {
        'train': PyGDataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        'val': PyGDataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        'test': PyGDataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Test")
    print("=" * 60)

    # Create test dataset
    from data_curation import DataCurationPipeline

    # Run curation first
    pipeline = DataCurationPipeline()
    splits = pipeline.run(use_cache=False)

    # Create dataset
    dataset = TransporterDataset(
        CONFIG.data.data_dir,
        split='train',
        use_3d=False,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"\nStatistics: {dataset.get_statistics()}")
    print(f"\nClass weights: {dataset.get_class_weights()}")

    if len(dataset) > 0:
        # Test loading a sample
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Node features: {sample.x.shape}")
        print(f"  Edge features: {sample.edge_attr.shape}")
        print(f"  DAT label: {sample.y_dat}")
        print(f"  NET label: {sample.y_net}")
        print(f"  SERT label: {sample.y_sert}")

        # Test dataloader
        dataloaders = create_dataloaders(
            CONFIG.data.data_dir,
            batch_size=4,
            num_workers=0,
            use_3d=False,
        )

        for batch in dataloaders['train']:
            print(f"\nBatch:")
            print(f"  Batch size: {batch.num_graphs}")
            print(f"  Node features: {batch.x.shape}")
            break
