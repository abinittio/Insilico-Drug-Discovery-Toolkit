"""
Comprehensive Transporter Data for Pretraining
===============================================

Fetch ALL transporter data from ChEMBL for pretraining:
- SLC6 family (monoamine, amino acid, GABA transporters)
- SLC22 family (OCT, OAT transporters)
- ABC transporters (P-gp, BCRP, MRP)
- SLC1 family (glutamate transporters)
- SLC7 family (amino acid transporters)

This provides 50k-100k+ compounds for pretraining,
then we fine-tune on monoamine-specific data.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logger.warning("ChEMBL client not available - will use cached/synthetic data")


class AllTransportersFetcher:
    """
    Fetch ALL transporter data from ChEMBL for pretraining.

    This includes:
    - SLC6: Monoamine + amino acid transporters
    - SLC22: Organic cation/anion transporters
    - ABC: ATP-binding cassette transporters
    - SLC1: Glutamate transporters
    - SLC7: Amino acid transporters
    """

    # Comprehensive transporter target list
    TRANSPORTER_FAMILIES = {
        # SLC6 - Neurotransmitter transporters (PRIMARY for monoamines)
        'SLC6': {
            'DAT': {'chembl_id': 'CHEMBL238', 'gene': 'SLC6A3', 'substrate_type': 'monoamine'},
            'NET': {'chembl_id': 'CHEMBL222', 'gene': 'SLC6A2', 'substrate_type': 'monoamine'},
            'SERT': {'chembl_id': 'CHEMBL228', 'gene': 'SLC6A4', 'substrate_type': 'monoamine'},
            'GAT1': {'chembl_id': 'CHEMBL1997', 'gene': 'SLC6A1', 'substrate_type': 'amino_acid'},
            'GAT2': {'chembl_id': 'CHEMBL5112', 'gene': 'SLC6A13', 'substrate_type': 'amino_acid'},
            'GAT3': {'chembl_id': 'CHEMBL2954', 'gene': 'SLC6A11', 'substrate_type': 'amino_acid'},
            'BGT1': {'chembl_id': 'CHEMBL5608', 'gene': 'SLC6A12', 'substrate_type': 'amino_acid'},
            'GlyT1': {'chembl_id': 'CHEMBL2018', 'gene': 'SLC6A9', 'substrate_type': 'amino_acid'},
            'GlyT2': {'chembl_id': 'CHEMBL3371', 'gene': 'SLC6A5', 'substrate_type': 'amino_acid'},
        },

        # SLC18 - Vesicular transporters
        'SLC18': {
            'VMAT1': {'chembl_id': 'CHEMBL1907', 'gene': 'SLC18A1', 'substrate_type': 'monoamine'},
            'VMAT2': {'chembl_id': 'CHEMBL4860', 'gene': 'SLC18A2', 'substrate_type': 'monoamine'},
            'VAChT': {'chembl_id': 'CHEMBL5979', 'gene': 'SLC18A3', 'substrate_type': 'other'},
        },

        # SLC22 - Organic cation/anion transporters (important for drug disposition)
        'SLC22': {
            'OCT1': {'chembl_id': 'CHEMBL1877', 'gene': 'SLC22A1', 'substrate_type': 'organic_cation'},
            'OCT2': {'chembl_id': 'CHEMBL1743122', 'gene': 'SLC22A2', 'substrate_type': 'organic_cation'},
            'OCT3': {'chembl_id': 'CHEMBL2364660', 'gene': 'SLC22A3', 'substrate_type': 'organic_cation'},
            'OAT1': {'chembl_id': 'CHEMBL1743125', 'gene': 'SLC22A6', 'substrate_type': 'organic_anion'},
            'OAT3': {'chembl_id': 'CHEMBL1743124', 'gene': 'SLC22A8', 'substrate_type': 'organic_anion'},
            'OCTN1': {'chembl_id': 'CHEMBL3885552', 'gene': 'SLC22A4', 'substrate_type': 'organic_cation'},
            'OCTN2': {'chembl_id': 'CHEMBL3885580', 'gene': 'SLC22A5', 'substrate_type': 'organic_cation'},
            'MATE1': {'chembl_id': 'CHEMBL3885613', 'gene': 'SLC47A1', 'substrate_type': 'organic_cation'},
            'MATE2K': {'chembl_id': 'CHEMBL3885581', 'gene': 'SLC47A2', 'substrate_type': 'organic_cation'},
        },

        # ABC transporters (efflux pumps - important for BBB/drug resistance)
        'ABC': {
            'P-gp': {'chembl_id': 'CHEMBL4302', 'gene': 'ABCB1', 'substrate_type': 'efflux'},
            'BCRP': {'chembl_id': 'CHEMBL5393', 'gene': 'ABCG2', 'substrate_type': 'efflux'},
            'MRP1': {'chembl_id': 'CHEMBL3620', 'gene': 'ABCC1', 'substrate_type': 'efflux'},
            'MRP2': {'chembl_id': 'CHEMBL3797', 'gene': 'ABCC2', 'substrate_type': 'efflux'},
            'MRP3': {'chembl_id': 'CHEMBL3797000', 'gene': 'ABCC3', 'substrate_type': 'efflux'},
        },

        # SLC1 - Glutamate transporters (important for CNS)
        'SLC1': {
            'EAAT1': {'chembl_id': 'CHEMBL4006', 'gene': 'SLC1A3', 'substrate_type': 'amino_acid'},
            'EAAT2': {'chembl_id': 'CHEMBL4007', 'gene': 'SLC1A2', 'substrate_type': 'amino_acid'},
            'EAAT3': {'chembl_id': 'CHEMBL4008', 'gene': 'SLC1A1', 'substrate_type': 'amino_acid'},
        },

        # SLCO/OATP - Organic anion transporting polypeptides
        'SLCO': {
            'OATP1A2': {'chembl_id': 'CHEMBL5621', 'gene': 'SLCO1A2', 'substrate_type': 'organic_anion'},
            'OATP1B1': {'chembl_id': 'CHEMBL1697677', 'gene': 'SLCO1B1', 'substrate_type': 'organic_anion'},
            'OATP1B3': {'chembl_id': 'CHEMBL1697678', 'gene': 'SLCO1B3', 'substrate_type': 'organic_anion'},
            'OATP2B1': {'chembl_id': 'CHEMBL1743123', 'gene': 'SLCO2B1', 'substrate_type': 'organic_anion'},
        },
    }

    # Keywords for substrate vs blocker classification
    SUBSTRATE_KEYWORDS = [
        'substrate', 'transport', 'uptake', 'efflux', 'release',
        'translocation', 'Km', 'Vmax', 'permeability', 'flux',
        'transcellular', 'transwell', 'caco-2', 'mdck',
    ]

    BLOCKER_KEYWORDS = [
        'inhibitor', 'inhibition', 'blocker', 'antagonist',
        'binding', 'displacement', 'Ki', 'IC50', 'competitive',
    ]

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("./data/transporter_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if CHEMBL_AVAILABLE:
            self.activity = new_client.activity
            self.molecule = new_client.molecule
        else:
            self.activity = None

    def fetch_target_data(
        self,
        target_name: str,
        target_info: Dict,
        family: str,
    ) -> pd.DataFrame:
        """Fetch all activity data for a single target."""
        cache_file = self.cache_dir / f"{family}_{target_name}_activities.parquet"

        if cache_file.exists():
            logger.info(f"Loading cached {target_name} data")
            return pd.read_parquet(cache_file)

        if not CHEMBL_AVAILABLE:
            logger.warning(f"ChEMBL not available, skipping {target_name}")
            return pd.DataFrame()

        logger.info(f"Fetching {target_name} ({target_info['chembl_id']}) from ChEMBL...")

        try:
            activities = self.activity.filter(
                target_chembl_id=target_info['chembl_id'],
            ).only([
                'molecule_chembl_id',
                'canonical_smiles',
                'standard_value',
                'standard_units',
                'standard_type',
                'pchembl_value',
                'assay_chembl_id',
                'assay_type',
                'assay_description',
            ])

            records = []
            for act in tqdm(activities, desc=f"Processing {target_name}"):
                smiles = act.get('canonical_smiles')
                if not smiles:
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Get potency
                pchembl = act.get('pchembl_value')
                if pchembl is None:
                    value = act.get('standard_value')
                    units = act.get('standard_units', '')
                    if value and value > 0:
                        if units in ['nM']:
                            pchembl = 9 - np.log10(value)
                        elif units in ['uM', 'µM']:
                            pchembl = 6 - np.log10(value)

                # Classify
                label, confidence = self._classify_activity(
                    assay_desc=act.get('assay_description', ''),
                    assay_type=act.get('assay_type', ''),
                    std_type=act.get('standard_type', ''),
                    pchembl=pchembl,
                )

                records.append({
                    'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
                    'target': target_name,
                    'target_family': family,
                    'substrate_type': target_info['substrate_type'],
                    'molecule_chembl_id': act.get('molecule_chembl_id'),
                    'pchembl': pchembl,
                    'standard_type': act.get('standard_type'),
                    'assay_type': act.get('assay_type'),
                    'assay_description': act.get('assay_description'),
                    'label': label,
                    'confidence': confidence,
                    'source': 'ChEMBL',
                })

            df = pd.DataFrame(records)
            logger.info(f"  Fetched {len(df)} records for {target_name}")

            if len(df) > 0:
                df.to_parquet(cache_file)

            return df

        except Exception as e:
            logger.error(f"Error fetching {target_name}: {e}")
            return pd.DataFrame()

    def _classify_activity(
        self,
        assay_desc: str,
        assay_type: str,
        std_type: str,
        pchembl: Optional[float],
    ) -> Tuple[int, float]:
        """
        Classify as substrate (2), blocker (1), inactive (0), or unknown (-1).
        """
        desc_lower = assay_desc.lower() if assay_desc else ""
        std_type_lower = std_type.lower() if std_type else ""

        is_active = pchembl is not None and pchembl >= 5.0

        has_substrate_kw = any(kw in desc_lower for kw in self.SUBSTRATE_KEYWORDS)
        has_blocker_kw = any(kw in desc_lower for kw in self.BLOCKER_KEYWORDS)

        # Explicit substrate assay
        if has_substrate_kw and not has_blocker_kw:
            if is_active:
                return 2, 0.80
            else:
                return 0, 0.65

        # Explicit blocker/inhibitor assay
        if has_blocker_kw and not has_substrate_kw:
            if is_active:
                return 1, 0.80
            else:
                return 0, 0.65

        # Binding assay - default to blocker
        if assay_type == 'B':
            if is_active:
                return 1, 0.60
            else:
                return 0, 0.55

        # Functional assay
        if assay_type == 'F':
            if std_type_lower in ['ic50', 'ki']:
                return (1, 0.70) if is_active else (0, 0.60)
            elif std_type_lower in ['ec50', 'km']:
                return (2, 0.65) if is_active else (0, 0.55)

        return -1, 0.30

    def fetch_all_families(
        self,
        families: List[str] = None,
        prioritize_monoamine: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch data for all transporter families.

        Args:
            families: Which families to fetch (None = all)
            prioritize_monoamine: Fetch SLC6/SLC18 first
        """
        all_dfs = []

        if families is None:
            families = list(self.TRANSPORTER_FAMILIES.keys())

        # Prioritize monoamine transporters
        if prioritize_monoamine:
            priority = ['SLC6', 'SLC18']
            families = [f for f in priority if f in families] + \
                      [f for f in families if f not in priority]

        for family in families:
            if family not in self.TRANSPORTER_FAMILIES:
                continue

            targets = self.TRANSPORTER_FAMILIES[family]
            logger.info(f"\n{'='*50}")
            logger.info(f"Fetching {family} family ({len(targets)} targets)")
            logger.info(f"{'='*50}")

            for target_name, target_info in targets.items():
                df = self.fetch_target_data(target_name, target_info, family)
                if len(df) > 0:
                    all_dfs.append(df)
                time.sleep(0.5)  # Rate limiting

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)

        # Remove unknown labels
        combined = combined[combined['label'] >= 0]

        return combined

    def aggregate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate multiple measurements per compound-target pair."""
        if len(df) == 0:
            return df

        aggregated = []

        for (smiles, target), group in df.groupby(['smiles', 'target']):
            votes = {0: 0, 1: 0, 2: 0}
            for _, row in group.iterrows():
                label = row['label']
                conf = row.get('confidence', 0.5)
                if label >= 0:
                    votes[label] += conf

            if sum(votes.values()) == 0:
                continue

            final_label = max(votes.keys(), key=lambda k: votes[k])
            total_conf = sum(votes.values())
            final_conf = votes[final_label] / total_conf if total_conf > 0 else 0.5

            aggregated.append({
                'smiles': smiles,
                'target': target,
                'target_family': group['target_family'].iloc[0],
                'substrate_type': group['substrate_type'].iloc[0],
                'label': final_label,
                'confidence': final_conf,
                'num_measurements': len(group),
                'source': 'ChEMBL',
            })

        return pd.DataFrame(aggregated)


def create_pretraining_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the full pretraining dataset.

    Returns:
        general_df: All transporter data for pretraining
        monoamine_df: DAT/NET/SERT specific data for fine-tuning
    """
    fetcher = AllTransportersFetcher()

    # Fetch all data
    logger.info("="*70)
    logger.info("FETCHING ALL TRANSPORTER DATA FOR PRETRAINING")
    logger.info("="*70)

    all_data = fetcher.fetch_all_families()

    if len(all_data) == 0:
        logger.warning("No data fetched - using fallback data generation")
        return _generate_fallback_data()

    # Aggregate
    aggregated = fetcher.aggregate_labels(all_data)

    # Split into general vs monoamine-specific
    monoamine_targets = ['DAT', 'NET', 'SERT', 'VMAT1', 'VMAT2']

    monoamine_df = aggregated[aggregated['target'].isin(monoamine_targets)]
    general_df = aggregated  # Keep all for pretraining

    # Statistics
    logger.info("\n" + "="*70)
    logger.info("PRETRAINING DATASET STATISTICS")
    logger.info("="*70)

    logger.info(f"\nGeneral (all transporters):")
    logger.info(f"  Total records: {len(general_df)}")
    logger.info(f"  Unique compounds: {general_df['smiles'].nunique()}")

    for family in general_df['target_family'].unique():
        f_df = general_df[general_df['target_family'] == family]
        logger.info(f"  {family}: {len(f_df)} records, {f_df['smiles'].nunique()} compounds")

    logger.info(f"\nMonoamine-specific (for fine-tuning):")
    logger.info(f"  Total records: {len(monoamine_df)}")
    logger.info(f"  Unique compounds: {monoamine_df['smiles'].nunique()}")

    for target in monoamine_targets:
        t_df = monoamine_df[monoamine_df['target'] == target]
        if len(t_df) > 0:
            subs = len(t_df[t_df['label'] == 2])
            block = len(t_df[t_df['label'] == 1])
            inact = len(t_df[t_df['label'] == 0])
            logger.info(f"  {target}: {subs} sub / {block} block / {inact} inact")

    return general_df, monoamine_df


def _generate_fallback_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic fallback data when ChEMBL is unavailable.
    Uses known compounds and structural analogs.
    """
    logger.info("Generating fallback pretraining data...")

    # Import from existing modules
    from data_comprehensive import ComprehensiveLiteratureData
    from data_sar_expansion import SARExpander, DecoyGenerator
    from data_additional import get_additional_data

    # Get all monoamine data
    lit_df = ComprehensiveLiteratureData.get_all_data()

    expander = SARExpander()
    sar_df = expander.generate_all()

    decoy_gen = DecoyGenerator()
    decoy_df = decoy_gen.generate_decoys()

    add_df = get_additional_data()

    monoamine_df = pd.concat([lit_df, sar_df, decoy_df, add_df], ignore_index=True)

    # Canonicalize
    def canonicalize(smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

    monoamine_df['smiles'] = monoamine_df['smiles'].apply(canonicalize)
    monoamine_df = monoamine_df[monoamine_df['smiles'].notna()]
    monoamine_df = monoamine_df.drop_duplicates(subset=['smiles', 'target'])

    # Add metadata columns
    monoamine_df['target_family'] = 'SLC6'
    monoamine_df['substrate_type'] = 'monoamine'

    # For pretraining without ChEMBL, just use monoamine data
    # In production, this would include all transporters
    general_df = monoamine_df.copy()

    logger.info(f"Fallback data: {len(monoamine_df)} monoamine records")

    return general_df, monoamine_df


def prepare_pretraining_splits(
    general_df: pd.DataFrame,
    monoamine_df: pd.DataFrame,
    output_dir: Path = None,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare train/val/test splits for pretraining and fine-tuning.

    Strategy:
    1. Pretrain on general transporter data (80/10/10 random split)
    2. Fine-tune on monoamine data (scaffold split for rigorous eval)
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    output_dir = output_dir or Path("./data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pretraining splits (random)
    logger.info("\nCreating pretraining splits (random)...")

    general_shuffled = general_df.sample(frac=1, random_state=42)
    n = len(general_shuffled)

    pretrain_train = general_shuffled.iloc[:int(0.8*n)]
    pretrain_val = general_shuffled.iloc[int(0.8*n):int(0.9*n)]
    pretrain_test = general_shuffled.iloc[int(0.9*n):]

    logger.info(f"  Pretrain train: {len(pretrain_train)}")
    logger.info(f"  Pretrain val: {len(pretrain_val)}")
    logger.info(f"  Pretrain test: {len(pretrain_test)}")

    # Fine-tuning splits (scaffold split)
    logger.info("\nCreating fine-tuning splits (scaffold)...")

    def get_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except:
            pass
        return smiles

    monoamine_df = monoamine_df.copy()
    monoamine_df['scaffold'] = monoamine_df['smiles'].apply(get_scaffold)

    scaffolds = monoamine_df['scaffold'].unique()
    np.random.seed(42)
    np.random.shuffle(scaffolds)

    n_scaffolds = len(scaffolds)
    train_scaffolds = set(scaffolds[:int(0.8*n_scaffolds)])
    val_scaffolds = set(scaffolds[int(0.8*n_scaffolds):int(0.9*n_scaffolds)])
    test_scaffolds = set(scaffolds[int(0.9*n_scaffolds):])

    finetune_train = monoamine_df[monoamine_df['scaffold'].isin(train_scaffolds)]
    finetune_val = monoamine_df[monoamine_df['scaffold'].isin(val_scaffolds)]
    finetune_test = monoamine_df[monoamine_df['scaffold'].isin(test_scaffolds)]

    logger.info(f"  Fine-tune train: {len(finetune_train)}")
    logger.info(f"  Fine-tune val: {len(finetune_val)}")
    logger.info(f"  Fine-tune test: {len(finetune_test)}")

    # Save
    splits = {
        'pretrain_train': pretrain_train,
        'pretrain_val': pretrain_val,
        'pretrain_test': pretrain_test,
        'finetune_train': finetune_train,
        'finetune_val': finetune_val,
        'finetune_test': finetune_test,
    }

    for name, df in splits.items():
        df.to_parquet(output_dir / f"{name}.parquet")
        logger.info(f"Saved {name} to {output_dir / f'{name}.parquet'}")

    # Also save combined datasets
    general_df.to_parquet(output_dir / "all_transporters.parquet")
    monoamine_df.to_parquet(output_dir / "monoamine_transporters.parquet")

    return splits


if __name__ == "__main__":
    # Create full pretraining dataset
    general_df, monoamine_df = create_pretraining_dataset()

    # Prepare splits
    splits = prepare_pretraining_splits(general_df, monoamine_df)

    print("\n" + "="*70)
    print("PRETRAINING DATA READY")
    print("="*70)
    print(f"General transporter compounds: {general_df['smiles'].nunique()}")
    print(f"Monoamine-specific compounds: {monoamine_df['smiles'].nunique()}")
