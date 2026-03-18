"""
Exhaustive ChEMBL Data Fetching
===============================

Fetch ALL available transporter data from ChEMBL:
- DAT: SLC6A3 (CHEMBL238)
- NET: SLC6A2 (CHEMBL222)
- SERT: SLC6A4 (CHEMBL228)

Also fetch:
- Vesicular monoamine transporters (VMAT1, VMAT2)
- Related SLC6 family members

Strategy:
1. Fetch all activity data (not just substrate assays)
2. Use pChEMBL >= 5 (10 uM) as activity threshold
3. Default to "blocker" for binding assays (conservative)
4. Only label as "substrate" if assay explicitly shows transport
"""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import time

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
    logger.warning("ChEMBL client not available")


class ExhaustiveChEMBLFetcher:
    """
    Exhaustively fetch all transporter data from ChEMBL.
    """

    # All relevant transporter targets
    TARGETS = {
        # Primary monoamine transporters
        'DAT': {'chembl_id': 'CHEMBL238', 'uniprot': 'Q01959', 'gene': 'SLC6A3'},
        'NET': {'chembl_id': 'CHEMBL222', 'uniprot': 'P23975', 'gene': 'SLC6A2'},
        'SERT': {'chembl_id': 'CHEMBL228', 'uniprot': 'P31645', 'gene': 'SLC6A4'},

        # Vesicular transporters (for additional context)
        'VMAT1': {'chembl_id': 'CHEMBL1907', 'uniprot': 'P54219', 'gene': 'SLC18A1'},
        'VMAT2': {'chembl_id': 'CHEMBL4860', 'uniprot': 'Q05940', 'gene': 'SLC18A2'},
    }

    # Keywords that STRONGLY indicate substrate behavior
    SUBSTRATE_KEYWORDS = [
        'release', 'releasing', 'efflux', 'reverse transport',
        'substrate', 'translocation', 'Km', 'Vmax',
        'superfusion', 'releaser', 'uptake substrate',
    ]

    # Keywords that indicate blocker/inhibitor (but NOT substrate)
    BLOCKER_KEYWORDS = [
        'reuptake inhibitor', 'uptake inhibitor', 'inhibition of uptake',
        'binding', 'displacement', 'radioligand', 'blocker',
        'competitive inhibitor', 'non-competitive',
    ]

    # Ambiguous - could be either
    AMBIGUOUS_KEYWORDS = [
        'uptake', 'IC50', 'Ki', 'transport',  # Need context
    ]

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("./data/chembl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if CHEMBL_AVAILABLE:
            self.activity = new_client.activity
            self.molecule = new_client.molecule
            self.assay = new_client.assay
            self.target = new_client.target
        else:
            self.activity = None

    def fetch_target(self, target_name: str, target_info: Dict) -> pd.DataFrame:
        """Fetch all activities for a single target."""
        cache_file = self.cache_dir / f"{target_name}_activities.parquet"

        if cache_file.exists():
            logger.info(f"Loading cached {target_name} data")
            return pd.read_parquet(cache_file)

        if not CHEMBL_AVAILABLE:
            return pd.DataFrame()

        logger.info(f"Fetching {target_name} from ChEMBL...")

        chembl_id = target_info['chembl_id']

        try:
            # Fetch all activities
            activities = self.activity.filter(
                target_chembl_id=chembl_id,
            ).only([
                'molecule_chembl_id',
                'canonical_smiles',
                'standard_value',
                'standard_units',
                'standard_type',
                'standard_relation',
                'pchembl_value',
                'assay_chembl_id',
                'assay_type',
                'assay_description',
                'target_chembl_id',
                'document_chembl_id',
            ])

            records = []
            for act in tqdm(activities, desc=f"Processing {target_name}"):
                smiles = act.get('canonical_smiles')
                if not smiles:
                    continue

                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Get pChEMBL value for potency
                pchembl = act.get('pchembl_value')
                if pchembl is None:
                    # Try to calculate from value
                    value = act.get('standard_value')
                    units = act.get('standard_units', '')
                    if value and value > 0:
                        # Convert to pChEMBL (-log10(M))
                        if units in ['nM']:
                            pchembl = 9 - np.log10(value)
                        elif units in ['uM', 'µM']:
                            pchembl = 6 - np.log10(value)
                        elif units in ['mM']:
                            pchembl = 3 - np.log10(value)

                # Classify activity
                label, confidence = self._classify_activity(
                    assay_desc=act.get('assay_description', ''),
                    assay_type=act.get('assay_type', ''),
                    std_type=act.get('standard_type', ''),
                    pchembl=pchembl,
                )

                records.append({
                    'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
                    'target': target_name,
                    'molecule_chembl_id': act.get('molecule_chembl_id'),
                    'pchembl': pchembl,
                    'standard_type': act.get('standard_type'),
                    'assay_type': act.get('assay_type'),
                    'assay_description': act.get('assay_description'),
                    'label': label,
                    'confidence': confidence,
                    'source': 'ChEMBL',
                    'category': 'chembl',
                })

            df = pd.DataFrame(records)
            logger.info(f"  Fetched {len(df)} records for {target_name}")

            # Cache
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
        Classify activity as substrate, blocker, or inactive.

        Returns:
            (label, confidence)
            label: 2=substrate, 1=blocker, 0=inactive, -1=unknown
        """
        desc_lower = assay_desc.lower() if assay_desc else ""
        std_type_lower = std_type.lower() if std_type else ""

        # Check if active (pChEMBL >= 5 = 10 uM)
        is_active = pchembl is not None and pchembl >= 5.0

        # Check for substrate keywords
        has_substrate_keyword = any(kw in desc_lower for kw in self.SUBSTRATE_KEYWORDS)

        # Check for blocker keywords
        has_blocker_keyword = any(kw in desc_lower for kw in self.BLOCKER_KEYWORDS)

        # Explicit substrate assay
        if has_substrate_keyword and not has_blocker_keyword:
            if is_active:
                return 2, 0.85  # Substrate
            else:
                return 0, 0.70  # Inactive

        # Explicit blocker assay
        if has_blocker_keyword and not has_substrate_keyword:
            if is_active:
                return 1, 0.85  # Blocker
            else:
                return 0, 0.70  # Inactive

        # Binding assays (B) - default to blocker if active
        if assay_type == 'B':
            if is_active:
                return 1, 0.60  # Likely blocker (binding doesn't mean transport)
            else:
                return 0, 0.60  # Inactive

        # Functional assays (F) - could be either
        if assay_type == 'F':
            # Check standard type
            if std_type_lower in ['ec50', 'emax']:
                # Functional response - could be substrate or blocker
                if is_active:
                    return -1, 0.40  # Ambiguous
            elif std_type_lower in ['ic50', 'ki']:
                # Inhibition - likely blocker
                if is_active:
                    return 1, 0.70
                else:
                    return 0, 0.65

        # Default: ambiguous
        return -1, 0.30

    def fetch_all(self) -> pd.DataFrame:
        """Fetch data for all targets."""
        all_dfs = []

        for target_name, target_info in self.TARGETS.items():
            df = self.fetch_target(target_name, target_info)
            if len(df) > 0:
                all_dfs.append(df)
            time.sleep(1)  # Rate limiting

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)

        # Remove ambiguous labels
        combined = combined[combined['label'] >= 0]

        logger.info(f"\nTotal ChEMBL data: {len(combined)} records")
        logger.info(f"Unique compounds: {combined['smiles'].nunique()}")

        return combined

    def aggregate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple measurements per compound-target pair.

        Uses confidence-weighted voting.
        """
        if len(df) == 0:
            return df

        aggregated = []

        for (smiles, target), group in df.groupby(['smiles', 'target']):
            # Weight by confidence
            votes = {0: 0, 1: 0, 2: 0}
            for _, row in group.iterrows():
                label = row['label']
                conf = row.get('confidence', 0.5)
                if label >= 0:
                    votes[label] += conf

            if sum(votes.values()) == 0:
                continue

            # Select highest weighted label
            final_label = max(votes.keys(), key=lambda k: votes[k])
            total_conf = sum(votes.values())
            final_conf = votes[final_label] / total_conf

            aggregated.append({
                'smiles': smiles,
                'target': target,
                'label': final_label,
                'confidence': final_conf,
                'num_measurements': len(group),
                'source': 'ChEMBL',
                'category': 'chembl_aggregated',
                'compound_name': group['molecule_chembl_id'].iloc[0],
            })

        return pd.DataFrame(aggregated)


def fetch_exhaustive_chembl() -> pd.DataFrame:
    """Convenience function to fetch all ChEMBL data."""
    fetcher = ExhaustiveChEMBLFetcher()
    df = fetcher.fetch_all()

    if len(df) > 0:
        df = fetcher.aggregate_labels(df)

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("EXHAUSTIVE CHEMBL STATISTICS")
    logger.info("=" * 60)

    if len(df) > 0:
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Unique compounds: {df['smiles'].nunique()}")

        for target in df['target'].unique():
            t_df = df[df['target'] == target]
            subs = len(t_df[t_df['label'] == 2])
            block = len(t_df[t_df['label'] == 1])
            inact = len(t_df[t_df['label'] == 0])
            logger.info(f"  {target}: {subs} sub / {block} block / {inact} inact")

    return df


if __name__ == "__main__":
    df = fetch_exhaustive_chembl()

    if len(df) > 0:
        output_path = Path("./data/chembl_exhaustive.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        print(f"\nSaved to {output_path}")
