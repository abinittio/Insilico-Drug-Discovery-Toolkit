"""
Unified Data Curation Pipeline
==============================

Integrates ALL data sources into a single curated dataset:

1. Comprehensive literature data (data_comprehensive.py)
2. SAR-expanded analogs (data_sar_expansion.py)
3. ChEMBL assay data (with rigorous filtering)
4. Scaffold-based train/val/test splitting

Target: 2000+ unique compounds with reliable labels
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

# Import our data sources
from data_comprehensive import ComprehensiveDataCurator, ComprehensiveLiteratureData
from data_sar_expansion import SARExpander, DecoyGenerator

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    print("ChEMBL client not available - using curated data only")

from config import CONFIG, DataConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ChEMBLFetcher:
    """
    Fetch transporter activity data from ChEMBL.

    Applies rigorous filtering to distinguish substrates from blockers.
    """

    TARGET_IDS = {
        'DAT': 'CHEMBL238',   # SLC6A3
        'NET': 'CHEMBL222',   # SLC6A2
        'SERT': 'CHEMBL228',  # SLC6A4
    }

    # Assay keywords indicating SUBSTRATE behavior
    SUBSTRATE_KEYWORDS = [
        'uptake', 'transport', 'release', 'efflux',
        'substrate', 'translocation', 'Km', 'Vmax',
        'superfusion', 'releaser',
    ]

    # Assay keywords indicating BLOCKER behavior
    BLOCKER_KEYWORDS = [
        'binding', 'Ki', 'inhibitor', 'inhibition',
        'displacement', 'radioligand', 'antagonist',
        'blocker', 'IC50', 'reuptake inhibit',
    ]

    def __init__(self):
        if CHEMBL_AVAILABLE:
            self.activity = new_client.activity
            self.assay = new_client.assay
        else:
            self.activity = None

    def fetch_all(self) -> pd.DataFrame:
        """Fetch data for all transporter targets."""
        if not CHEMBL_AVAILABLE:
            logger.warning("ChEMBL not available")
            return pd.DataFrame()

        all_records = []

        for target_name, target_id in self.TARGET_IDS.items():
            logger.info(f"Fetching ChEMBL data for {target_name}...")

            try:
                activities = self.activity.filter(
                    target_chembl_id=target_id,
                    standard_type__in=['IC50', 'Ki', 'EC50', 'Activity', 'Inhibition'],
                ).only([
                    'molecule_chembl_id',
                    'canonical_smiles',
                    'standard_value',
                    'standard_units',
                    'standard_type',
                    'assay_description',
                    'pchembl_value',
                ])

                for act in tqdm(activities, desc=f"Processing {target_name}"):
                    smiles = act.get('canonical_smiles')
                    if not smiles:
                        continue

                    # Classify based on assay description
                    desc = (act.get('assay_description') or '').lower()
                    label, confidence = self._classify_assay(
                        desc,
                        act.get('standard_value'),
                        act.get('standard_type'),
                        act.get('standard_units'),
                    )

                    if label < 0:  # Unknown/ambiguous
                        continue

                    all_records.append({
                        'smiles': smiles,
                        'target': target_name,
                        'label': label,
                        'confidence': confidence,
                        'source': 'ChEMBL',
                        'category': 'chembl',
                        'compound_name': act.get('molecule_chembl_id'),
                    })

            except Exception as e:
                logger.error(f"Error fetching {target_name}: {e}")

        df = pd.DataFrame(all_records)
        logger.info(f"Fetched {len(df)} ChEMBL records")
        return df

    def _classify_assay(
        self,
        description: str,
        value: float,
        std_type: str,
        units: str,
    ) -> Tuple[int, float]:
        """
        Classify compound based on assay description.

        Returns:
            (label, confidence)
            label: 2=substrate, 1=blocker, 0=inactive, -1=unknown
        """
        # Check for substrate keywords
        is_substrate_assay = any(kw in description for kw in self.SUBSTRATE_KEYWORDS)
        is_blocker_assay = any(kw in description for kw in self.BLOCKER_KEYWORDS)

        # If explicitly described as substrate assay
        if is_substrate_assay and not is_blocker_assay:
            if self._is_active(value, std_type, units):
                return 2, 0.75  # Substrate
            else:
                return 0, 0.60  # Inactive

        # If explicitly described as blocker/binding assay
        if is_blocker_assay and not is_substrate_assay:
            if self._is_active(value, std_type, units):
                return 1, 0.75  # Blocker
            else:
                return 0, 0.60  # Inactive

        # Ambiguous
        return -1, 0.0

    def _is_active(self, value: float, std_type: str, units: str) -> bool:
        """Check if activity value indicates active compound."""
        if value is None:
            return False

        # Normalize to uM
        if units in ['nM', 'nmol/L']:
            value_um = value / 1000
        elif units in ['uM', 'umol/L', 'µM']:
            value_um = value
        elif units in ['mM']:
            value_um = value * 1000
        else:
            value_um = value

        # Activity threshold: < 10 uM
        return value_um < 10.0


class MolecularFilter:
    """Filter molecules by druglike properties."""

    def __init__(self, config: DataConfig = None):
        self.config = config or CONFIG.data

    def filter(self, smiles: str) -> bool:
        """Check if molecule passes all filters."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # MW filter
        mw = Descriptors.MolWt(mol)
        if mw < self.config.min_molecular_weight or mw > self.config.max_molecular_weight:
            return False

        # Heavy atom count
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.config.min_heavy_atoms or n_heavy > self.config.max_heavy_atoms:
            return False

        # No metals
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [3, 11, 12, 19, 20, 26, 29, 30]:
                return False

        return True


class ScaffoldSplitter:
    """Scaffold-based data splitting."""

    def __init__(self, config: DataConfig = None):
        self.config = config or CONFIG.data

    def get_scaffold(self, smiles: str) -> str:
        """Get Murcko scaffold."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return ""

    def split(
        self,
        smiles_list: List[str],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split by scaffolds.

        Returns:
            (train_indices, val_indices, test_indices)
        """
        # Group by scaffold
        scaffold_to_idx = defaultdict(list)
        for i, smi in enumerate(smiles_list):
            scaffold = self.get_scaffold(smi)
            scaffold_to_idx[scaffold].append(i)

        # Sort scaffolds by size
        scaffolds = sorted(
            scaffold_to_idx.keys(),
            key=lambda s: len(scaffold_to_idx[s]),
            reverse=True
        )

        # Shuffle with seed
        np.random.seed(self.config.scaffold_split_seed)
        np.random.shuffle(scaffolds)

        # Allocate to splits
        n_total = len(smiles_list)
        n_train = int((1 - self.config.val_fraction - self.config.test_fraction) * n_total)
        n_val = int(self.config.val_fraction * n_total)

        train_idx, val_idx, test_idx = [], [], []

        for scaffold in scaffolds:
            indices = scaffold_to_idx[scaffold]

            if len(train_idx) < n_train:
                train_idx.extend(indices)
            elif len(val_idx) < n_val:
                val_idx.extend(indices)
            else:
                test_idx.extend(indices)

        return train_idx, val_idx, test_idx


class UnifiedDataCurator:
    """
    Master curator that combines all data sources.
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or CONFIG.data
        self.filter = MolecularFilter(self.config)
        self.splitter = ScaffoldSplitter(self.config)

    def curate(self) -> pd.DataFrame:
        """
        Run complete curation pipeline.

        Returns:
            Curated DataFrame
        """
        logger.info("=" * 70)
        logger.info("UNIFIED DATA CURATION PIPELINE")
        logger.info("=" * 70)

        all_dfs = []

        # 1. Literature data (highest priority)
        logger.info("\n[1/4] Loading comprehensive literature data...")
        lit_curator = ComprehensiveDataCurator()
        lit_df = lit_curator.curate()
        lit_df['priority'] = 1  # Highest
        all_dfs.append(lit_df)
        logger.info(f"  Literature: {len(lit_df)} records")

        # 2. SAR-expanded analogs
        logger.info("\n[2/4] Generating SAR-expanded analogs...")
        sar_expander = SARExpander()
        sar_df = sar_expander.generate_all()
        sar_df['priority'] = 2
        all_dfs.append(sar_df)
        logger.info(f"  SAR-expanded: {len(sar_df)} records")

        # 3. Decoys
        logger.info("\n[3/4] Generating decoy compounds...")
        decoy_gen = DecoyGenerator()
        decoy_df = decoy_gen.generate_decoys()
        decoy_df['priority'] = 3
        all_dfs.append(decoy_df)
        logger.info(f"  Decoys: {len(decoy_df)} records")

        # 4. ChEMBL data (if available)
        logger.info("\n[4/4] Fetching ChEMBL data...")
        if CHEMBL_AVAILABLE:
            chembl = ChEMBLFetcher()
            chembl_df = chembl.fetch_all()
            if len(chembl_df) > 0:
                chembl_df['priority'] = 4  # Lowest priority
                all_dfs.append(chembl_df)
                logger.info(f"  ChEMBL: {len(chembl_df)} records")
        else:
            logger.info("  ChEMBL: Skipped (not available)")

        # Combine all
        df = pd.concat(all_dfs, ignore_index=True)

        # Apply molecular filter
        logger.info("\nFiltering molecules...")
        valid_mask = df['smiles'].apply(self.filter.filter)
        df = df[valid_mask].copy()
        logger.info(f"  After filtering: {len(df)} records")

        # Canonicalize SMILES
        logger.info("Canonicalizing SMILES...")
        df['smiles'] = df['smiles'].apply(self._canonicalize)
        df = df[df['smiles'] != ''].copy()

        # Deduplicate - keep highest priority (lowest number)
        logger.info("Deduplicating...")
        df = df.sort_values('priority')
        df = df.drop_duplicates(subset=['smiles', 'target'], keep='first')
        df = df.drop(columns=['priority'])

        # Add stereochemistry info
        logger.info("Extracting stereochemistry info...")
        stereo_info = df['smiles'].apply(self._get_stereo_info)
        df['num_stereocenters'] = stereo_info.apply(lambda x: x[0])
        df['has_stereocenters'] = stereo_info.apply(lambda x: x[1])
        df['stereo_defined'] = stereo_info.apply(lambda x: x[2])

        # Final statistics
        self._print_statistics(df)

        return df

    def _canonicalize(self, smiles: str) -> str:
        """Canonicalize SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ''
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            return ''

    def _get_stereo_info(self, smiles: str) -> Tuple[int, bool, bool]:
        """Get stereochemistry info."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0, False, True

        chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        num_centers = len(chiral)
        has_stereo = num_centers > 0
        defined = not any(c[1] == '?' for c in chiral)

        return num_centers, has_stereo, defined

    def _print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL DATASET STATISTICS")
        logger.info("=" * 70)

        logger.info(f"\nTotal records: {len(df)}")
        logger.info(f"Unique compounds: {df['smiles'].nunique()}")

        logger.info("\nPer-target breakdown:")
        for target in ['DAT', 'NET', 'SERT']:
            t_df = df[df['target'] == target]
            subs = len(t_df[t_df['label'] == 2])
            block = len(t_df[t_df['label'] == 1])
            inact = len(t_df[t_df['label'] == 0])
            total = len(t_df)
            logger.info(f"  {target}: {total:4d} total | {subs:3d} substrates | {block:3d} blockers | {inact:3d} inactive")

        logger.info("\nBy source:")
        for source in df['source'].unique():
            count = len(df[df['source'] == source])
            logger.info(f"  {source}: {count}")

        logger.info("\nStereochemistry:")
        stereo_compounds = df[df['has_stereocenters'] == True]['smiles'].nunique()
        total_compounds = df['smiles'].nunique()
        logger.info(f"  Compounds with stereocenters: {stereo_compounds}/{total_compounds} ({100*stereo_compounds/total_compounds:.1f}%)")

        defined = df[df['stereo_defined'] == True]['smiles'].nunique()
        logger.info(f"  Stereochemistry defined: {defined}/{stereo_compounds}")

    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/val/test.

        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        unique_smiles = df['smiles'].unique().tolist()

        train_idx, val_idx, test_idx = self.splitter.split(unique_smiles)

        train_smiles = set(np.array(unique_smiles)[train_idx])
        val_smiles = set(np.array(unique_smiles)[val_idx])
        test_smiles = set(np.array(unique_smiles)[test_idx])

        splits = {
            'train': df[df['smiles'].isin(train_smiles)].copy(),
            'val': df[df['smiles'].isin(val_smiles)].copy(),
            'test': df[df['smiles'].isin(test_smiles)].copy(),
        }

        logger.info("\nData splits:")
        for name, split_df in splits.items():
            n_compounds = split_df['smiles'].nunique()
            n_records = len(split_df)
            logger.info(f"  {name}: {n_compounds} compounds, {n_records} records")

        return splits

    def save(self, df: pd.DataFrame, splits: Dict[str, pd.DataFrame]):
        """Save curated data."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Save full dataset
        df.to_parquet(self.config.data_dir / "full_dataset.parquet")
        df.to_csv(self.config.data_dir / "full_dataset.csv", index=False)

        # Save splits
        for name, split_df in splits.items():
            split_df.to_parquet(self.config.data_dir / f"{name}.parquet")

        logger.info(f"\nData saved to {self.config.data_dir}")


class DataCurationPipeline:
    """
    Main pipeline class (drop-in replacement for original).
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or CONFIG.data
        self.curator = UnifiedDataCurator(self.config)

    def run(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run the complete pipeline.

        Args:
            use_cache: Use cached data if available

        Returns:
            Dict with 'train', 'val', 'test' DataFrames
        """
        cache_path = self.config.data_dir / "full_dataset.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            df = self.curator.curate()

        splits = self.curator.split_data(df)
        self.curator.save(df, splits)

        return splits

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_compounds': df['smiles'].nunique(),
            'total_records': len(df),
        }

        for target in ['DAT', 'NET', 'SERT']:
            t_df = df[df['target'] == target]
            stats[f'{target}_total'] = len(t_df)
            stats[f'{target}_substrates'] = len(t_df[t_df['label'] == 2])
            stats[f'{target}_blockers'] = len(t_df[t_df['label'] == 1])
            stats[f'{target}_inactive'] = len(t_df[t_df['label'] == 0])

        return stats


def main():
    """Run the full data curation pipeline."""
    print("=" * 70)
    print("MONOAMINE TRANSPORTER DATA CURATION")
    print("=" * 70)

    pipeline = DataCurationPipeline()
    splits = pipeline.run(use_cache=False)

    print("\n" + "=" * 70)
    print("CURATION COMPLETE")
    print("=" * 70)

    for name, df in splits.items():
        stats = pipeline.get_statistics(df)
        print(f"\n{name.upper()}:")
        print(f"  Compounds: {stats['total_compounds']}")
        for target in ['DAT', 'NET', 'SERT']:
            print(f"  {target}: {stats[f'{target}_substrates']} sub / {stats[f'{target}_blockers']} block / {stats[f'{target}_inactive']} inact")


if __name__ == "__main__":
    main()
