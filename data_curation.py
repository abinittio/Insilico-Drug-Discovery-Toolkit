"""
Data Curation Pipeline for Monoamine Transporter Substrates
============================================================

CRITICAL DISTINCTION: Substrate vs Blocker
------------------------------------------
This is the most important aspect of the data curation.

SUBSTRATE: A compound that is TRANSPORTED by the transporter
- Causes efflux (reverses transport direction)
- Is taken up into cells via the transporter
- Examples: Amphetamines, cathinones, MDMA

BLOCKER: A compound that INHIBITS transport without being transported
- Binds to transporter but is not translocated
- Blocks uptake of endogenous substrates
- Examples: Cocaine, methylphenidate, GBR 12909

The key insight: Ki/IC50 assays measure BINDING, not TRANSPORT.
We need UPTAKE/EFFLUX assays to distinguish substrates from blockers.

Data Sources:
1. ChEMBL - with careful assay type filtering
2. Literature curation - published substrate/blocker classifications
3. PDSP Ki database (for negative examples - blockers)
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logging.warning("ChEMBL client not available. Using cached/manual data.")

from config import CONFIG, DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompoundRecord:
    """A single compound with transporter activity data."""
    smiles: str
    canonical_smiles: str
    inchikey: str
    compound_name: Optional[str]

    # Activity labels (ternary: substrate=2, blocker=1, inactive=0, unknown=-1)
    dat_label: int = -1
    net_label: int = -1
    sert_label: int = -1

    # Continuous activity values if available
    dat_activity: Optional[float] = None
    net_activity: Optional[float] = None
    sert_activity: Optional[float] = None

    # Confidence in label (based on data quality)
    dat_confidence: float = 0.0
    net_confidence: float = 0.0
    sert_confidence: float = 0.0

    # Stereochemistry info
    has_stereocenters: bool = False
    num_stereocenters: int = 0
    is_racemic: bool = False
    stereo_defined: bool = True

    # Source tracking
    sources: List[str] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []


class SubstrateBlockerClassifier:
    """
    Classifies compounds as substrates vs blockers based on assay data.

    This is the CRITICAL component that distinguishes our approach.
    """

    # Assay types that indicate SUBSTRATE behavior
    SUBSTRATE_ASSAYS = {
        'uptake': 0.9,          # Direct transport measurement
        'efflux': 1.0,          # Substrate-induced efflux
        'release': 1.0,         # Releasing agent assay
        'translocation': 0.9,
        'transport': 0.9,
        'Km': 1.0,              # Michaelis-Menten kinetics = substrate
        'Vmax': 1.0,
        'superfusion': 0.8,     # Often used for release assays
    }

    # Assay types that indicate BLOCKER behavior (or are ambiguous)
    BLOCKER_ASSAYS = {
        'binding': 0.7,         # Binding doesn't mean transport
        'Ki': 0.8,              # Inhibition constant
        'IC50': 0.6,            # Could be either, lower confidence
        'displacement': 0.8,
        'radioligand': 0.7,
        'competition': 0.7,
    }

    # Keywords suggesting the compound IS a substrate in the assay description
    SUBSTRATE_KEYWORDS = [
        'substrate',
        'transported',
        'releaser',
        'releasing agent',
        'efflux',
        'reverse transport',
        'DAT substrate',
        'NET substrate',
        'SERT substrate',
    ]

    # Keywords suggesting the compound is a BLOCKER
    BLOCKER_KEYWORDS = [
        'blocker',
        'inhibitor',
        'uptake inhibitor',
        'reuptake inhibitor',
        'antagonist',
        'non-substrate',
        'competitive inhibitor',
    ]

    def classify_from_assay(
        self,
        assay_type: str,
        assay_description: str,
        activity_value: float,
        activity_type: str,
        standard_units: str,
    ) -> Tuple[int, float]:
        """
        Classify a compound based on a single assay result.

        Returns:
            Tuple of (label, confidence)
            label: 2=substrate, 1=blocker, 0=inactive, -1=unknown
        """
        assay_lower = assay_type.lower() if assay_type else ""
        desc_lower = assay_description.lower() if assay_description else ""

        # Check for explicit substrate/blocker keywords in description
        for kw in self.SUBSTRATE_KEYWORDS:
            if kw in desc_lower:
                # High confidence substrate
                if self._is_active(activity_value, activity_type, standard_units):
                    return 2, 0.95
                else:
                    return 0, 0.7

        for kw in self.BLOCKER_KEYWORDS:
            if kw in desc_lower:
                if self._is_active(activity_value, activity_type, standard_units):
                    return 1, 0.90
                else:
                    return 0, 0.7

        # Check assay type
        for substrate_assay, conf in self.SUBSTRATE_ASSAYS.items():
            if substrate_assay in assay_lower or substrate_assay in desc_lower:
                if self._is_active(activity_value, activity_type, standard_units):
                    return 2, conf
                else:
                    return 0, conf * 0.8

        for blocker_assay, conf in self.BLOCKER_ASSAYS.items():
            if blocker_assay in assay_lower or blocker_assay in desc_lower:
                if self._is_active(activity_value, activity_type, standard_units):
                    return 1, conf
                else:
                    return 0, conf * 0.8

        # Unknown assay type
        return -1, 0.0

    def _is_active(
        self,
        value: float,
        activity_type: str,
        units: str
    ) -> bool:
        """Determine if the activity value indicates an active compound."""
        if value is None:
            return False

        # Convert value to float if it's a string
        try:
            value = float(value)
        except (ValueError, TypeError):
            return False

        # Convert to comparable units
        if units in ['nM', 'nmol/L']:
            value_um = value / 1000
        elif units in ['uM', 'umol/L', 'µM']:
            value_um = value
        elif units in ['mM', 'mmol/L']:
            value_um = value * 1000
        elif units == '%':
            # Percentage inhibition/activity
            return value > 50  # >50% effect = active
        else:
            value_um = value  # Assume uM

        # For potency measures (Ki, IC50, EC50, etc.)
        if activity_type in ['Ki', 'IC50', 'EC50', 'Kd']:
            return value_um < 10.0  # < 10 uM = active

        # For efficacy measures
        if activity_type in ['Emax', 'Activity', 'Inhibition']:
            return value_um > 50  # Usually percentage

        return value_um < 10.0  # Default threshold


class ChEMBLDataFetcher:
    """Fetches and processes data from ChEMBL."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.classifier = SubstrateBlockerClassifier()

        if CHEMBL_AVAILABLE:
            self.activity = new_client.activity
            self.molecule = new_client.molecule
            self.target = new_client.target
        else:
            self.activity = None

    def fetch_target_data(self, target_name: str, target_chembl_id: str) -> pd.DataFrame:
        """
        Fetch all activity data for a transporter target.

        Args:
            target_name: Human readable name (DAT, NET, SERT)
            target_chembl_id: ChEMBL ID (e.g., CHEMBL238)

        Returns:
            DataFrame with processed activity data
        """
        logger.info(f"Fetching data for {target_name} ({target_chembl_id})")

        if not CHEMBL_AVAILABLE:
            logger.warning("ChEMBL not available, returning empty DataFrame")
            return pd.DataFrame()

        # Fetch all activities for this target
        activities = self.activity.filter(
            target_chembl_id=target_chembl_id,
            assay_type__in=['B', 'F'],  # Binding and Functional assays
        ).only([
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_value',
            'standard_units',
            'standard_type',
            'pchembl_value',
            'assay_type',
            'assay_description',
            'target_chembl_id',
            'activity_comment',
        ])

        records = []
        for act in tqdm(activities, desc=f"Processing {target_name}"):
            if not act.get('canonical_smiles'):
                continue

            # Classify this assay result
            label, confidence = self.classifier.classify_from_assay(
                assay_type=act.get('assay_type', ''),
                assay_description=act.get('assay_description', ''),
                activity_value=act.get('standard_value'),
                activity_type=act.get('standard_type', ''),
                standard_units=act.get('standard_units', ''),
            )

            records.append({
                'molecule_chembl_id': act.get('molecule_chembl_id'),
                'smiles': act.get('canonical_smiles'),
                'standard_value': act.get('standard_value'),
                'standard_units': act.get('standard_units'),
                'standard_type': act.get('standard_type'),
                'pchembl_value': act.get('pchembl_value'),
                'assay_type': act.get('assay_type'),
                'assay_description': act.get('assay_description'),
                'target': target_name,
                'label': label,
                'confidence': confidence,
                'activity_comment': act.get('activity_comment'),
            })

        df = pd.DataFrame(records)
        logger.info(f"Fetched {len(df)} records for {target_name}")
        return df

    def fetch_all_targets(self) -> pd.DataFrame:
        """Fetch data for all monoamine transporters."""
        all_dfs = []

        for target_name, chembl_id in self.config.chembl_targets.items():
            df = self.fetch_target_data(target_name, chembl_id)
            if len(df) > 0:
                all_dfs.append(df)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()


class LiteratureCurator:
    """
    Curates high-confidence substrate/blocker labels from literature.

    This provides our gold-standard labels that override ChEMBL data.
    """

    # KNOWN SUBSTRATES - these are DEFINITELY transported
    # Format: (SMILES, name, {DAT: label, NET: label, SERT: label})
    # Label: 2=substrate, 1=blocker, 0=inactive
    KNOWN_SUBSTRATES = [
        # Endogenous monoamines (the actual substrates)
        ("NCCc1ccc(O)c(O)c1", "Dopamine", {"DAT": 2, "NET": 2, "SERT": 0}),
        ("NC[C@H](O)c1ccc(O)c(O)c1", "(-)-Norepinephrine", {"DAT": 1, "NET": 2, "SERT": 0}),
        ("NCCc1c[nH]c2ccc(O)cc12", "Serotonin", {"DAT": 0, "NET": 0, "SERT": 2}),

        # Amphetamines
        ("C[C@H](N)Cc1ccccc1", "(+)-Amphetamine (d)", {"DAT": 2, "NET": 2, "SERT": 1}),
        ("C[C@@H](N)Cc1ccccc1", "(-)-Amphetamine (l)", {"DAT": 1, "NET": 1, "SERT": 1}),
        ("C[C@H](NC)Cc1ccccc1", "(+)-Methamphetamine (d)", {"DAT": 2, "NET": 2, "SERT": 1}),
        ("C[C@@H](NC)Cc1ccccc1", "(-)-Methamphetamine (l)", {"DAT": 1, "NET": 1, "SERT": 0}),

        # MDMA and analogs
        ("C[C@H](NC)Cc1ccc2OCOc2c1", "(+)-MDMA (S)", {"DAT": 2, "NET": 2, "SERT": 2}),
        ("C[C@@H](NC)Cc1ccc2OCOc2c1", "(-)-MDMA (R)", {"DAT": 1, "NET": 1, "SERT": 2}),
        ("C[C@H](N)Cc1ccc2OCOc2c1", "(+)-MDA (S)", {"DAT": 2, "NET": 2, "SERT": 2}),

        # Cathinones
        ("CC(N)C(=O)c1ccccc1", "Cathinone", {"DAT": 2, "NET": 2, "SERT": 1}),
        ("CNC(C)C(=O)c1ccccc1", "Methcathinone", {"DAT": 2, "NET": 2, "SERT": 1}),
        ("CCCC(NC)C(=O)c1ccccc1", "Bupropion", {"DAT": 2, "NET": 2, "SERT": 0}),
        ("CC(NC)C(=O)c1ccc2OCOc2c1", "Methylone", {"DAT": 2, "NET": 2, "SERT": 2}),
        ("CC(NC)C(=O)c1ccc2c(c1)OCO2", "MDPV-related", {"DAT": 2, "NET": 1, "SERT": 1}),

        # Phenethylamines
        ("NCCc1ccccc1", "Phenethylamine", {"DAT": 2, "NET": 2, "SERT": 1}),
        ("CNCCc1ccccc1", "N-Methylphenethylamine", {"DAT": 2, "NET": 2, "SERT": 1}),
        ("NCCc1ccc(O)cc1", "Tyramine", {"DAT": 2, "NET": 2, "SERT": 0}),
        ("CNCCc1ccc(O)cc1", "N-Methyltyramine", {"DAT": 2, "NET": 2, "SERT": 0}),

        # Tryptamines (SERT substrates)
        ("CNCCc1c[nH]c2ccccc12", "N-Methyltryptamine", {"DAT": 0, "NET": 0, "SERT": 2}),
        ("CN(C)CCc1c[nH]c2ccccc12", "DMT", {"DAT": 0, "NET": 0, "SERT": 2}),

        # Para-halogenated amphetamines
        ("C[C@H](N)Cc1ccc(Cl)cc1", "4-Chloroamphetamine", {"DAT": 2, "NET": 2, "SERT": 2}),
        ("C[C@H](N)Cc1ccc(F)cc1", "4-Fluoroamphetamine", {"DAT": 2, "NET": 2, "SERT": 2}),

        # MPP+ and neurotoxins (DAT substrates, used in research)
        # Note: MPP+ is a known DAT substrate used in Parkinson's research
    ]

    # KNOWN BLOCKERS - these inhibit but are NOT transported
    KNOWN_BLOCKERS = [
        # Cocaine and tropanes
        ("COC(=O)[C@H]1C[C@@H]2CC[C@H](C1)N2C", "Cocaine-core", {"DAT": 1, "NET": 1, "SERT": 1}),
        ("COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3", "Cocaine", {"DAT": 1, "NET": 1, "SERT": 1}),

        # GBR compounds (selective DAT blockers)
        ("Fc1ccc(C(c2ccc(F)cc2)N3CCCC3)cc1", "GBR-12909-core", {"DAT": 1, "NET": 0, "SERT": 0}),

        # SSRIs (SERT blockers)
        ("CNCCC(Oc1ccc(C(F)(F)F)cc1)c2ccccc2", "Fluoxetine", {"DAT": 0, "NET": 0, "SERT": 1}),
        ("CN(C)CCC(c1ccc(Cl)cc1)c2ccccc2", "Chlorpheniramine-like", {"DAT": 0, "NET": 0, "SERT": 1}),
        ("Fc1ccc(C(=O)CCCN2CCCCC2)cc1", "Citalopram-core", {"DAT": 0, "NET": 0, "SERT": 1}),

        # NET blockers
        ("CNC[C@H](c1ccccc1)c2ccccc2O", "Atomoxetine", {"DAT": 0, "NET": 1, "SERT": 0}),

        # Tricyclic compounds
        ("CN(C)CCCN1c2ccccc2CCc3ccccc13", "Imipramine-core", {"DAT": 0, "NET": 1, "SERT": 1}),

        # Methylphenidate (complex - blocker at DAT, weak substrate at high conc)
        ("COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2", "d-threo-Methylphenidate", {"DAT": 1, "NET": 1, "SERT": 0}),
        ("COC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2", "l-threo-Methylphenidate", {"DAT": 1, "NET": 1, "SERT": 0}),

        # Modafinil
        ("NC(=O)CS(=O)C(c1ccccc1)c2ccccc2", "Modafinil", {"DAT": 1, "NET": 0, "SERT": 0}),

        # Bupropion is interesting - substrate at DAT/NET but clinical doses = blocker effect
        # We classify it as substrate based on mechanism
    ]

    # INACTIVE compounds (neither substrate nor blocker at relevant concentrations)
    KNOWN_INACTIVE = [
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin", {"DAT": 0, "NET": 0, "SERT": 0}),
        ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Ibuprofen", {"DAT": 0, "NET": 0, "SERT": 0}),
        ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Caffeine", {"DAT": 0, "NET": 0, "SERT": 0}),
    ]

    def get_curated_data(self) -> pd.DataFrame:
        """Generate curated dataset from literature knowledge."""
        records = []

        # Process known substrates
        for smiles, name, labels in self.KNOWN_SUBSTRATES:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES for {name}: {smiles}")
                continue

            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            for target, label in labels.items():
                records.append({
                    'smiles': canonical,
                    'compound_name': name,
                    'target': target,
                    'label': label,
                    'confidence': 1.0,  # Literature-curated = highest confidence
                    'source': 'literature_substrate',
                })

        # Process known blockers
        for smiles, name, labels in self.KNOWN_BLOCKERS:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            for target, label in labels.items():
                records.append({
                    'smiles': canonical,
                    'compound_name': name,
                    'target': target,
                    'label': label,
                    'confidence': 1.0,
                    'source': 'literature_blocker',
                })

        # Process inactive
        for smiles, name, labels in self.KNOWN_INACTIVE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            for target, label in labels.items():
                records.append({
                    'smiles': canonical,
                    'compound_name': name,
                    'target': target,
                    'label': label,
                    'confidence': 1.0,
                    'source': 'literature_inactive',
                })

        df = pd.DataFrame(records)
        logger.info(f"Curated {len(df)} literature records")
        return df


class MolecularFilter:
    """Filters molecules based on structural properties."""

    def __init__(self, config: DataConfig):
        self.config = config

    def filter_molecule(self, smiles: str) -> Tuple[bool, str]:
        """
        Check if molecule passes all filters.

        Returns:
            Tuple of (passes, reason_if_failed)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES"

        # Molecular weight
        mw = Descriptors.MolWt(mol)
        if mw < self.config.min_molecular_weight:
            return False, f"MW too low: {mw:.1f}"
        if mw > self.config.max_molecular_weight:
            return False, f"MW too high: {mw:.1f}"

        # Heavy atoms
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < self.config.min_heavy_atoms:
            return False, f"Too few heavy atoms: {n_heavy}"
        if n_heavy > self.config.max_heavy_atoms:
            return False, f"Too many heavy atoms: {n_heavy}"

        # Check for problematic groups
        # (metals, very reactive groups, etc.)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in [3, 11, 19, 37, 55]:  # Alkali metals
                return False, "Contains alkali metal"
            if atom.GetAtomicNum() in [4, 12, 20, 38, 56]:  # Alkaline earth
                return False, "Contains alkaline earth metal"

        return True, ""

    def get_stereochemistry_info(self, smiles: str) -> Dict:
        """Extract stereochemistry information from molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        # Find stereocenters
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

        # Check for E/Z double bonds
        stereo_bonds = []
        for bond in mol.GetBonds():
            stereo = bond.GetStereo()
            if stereo != Chem.BondStereo.STEREONONE:
                stereo_bonds.append({
                    'bond_idx': bond.GetIdx(),
                    'stereo': str(stereo),
                })

        # Check if stereochemistry is fully defined
        has_undefined = any(c[1] == '?' for c in chiral_centers)

        return {
            'num_stereocenters': len(chiral_centers),
            'stereocenters': chiral_centers,
            'num_stereo_bonds': len(stereo_bonds),
            'stereo_bonds': stereo_bonds,
            'stereo_defined': not has_undefined,
            'has_stereocenters': len(chiral_centers) > 0 or len(stereo_bonds) > 0,
        }


class ScaffoldSplitter:
    """
    Implements scaffold-based train/val/test splitting.

    This is CRITICAL for proper evaluation - ensures we test on
    novel scaffolds, not just novel substituents.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def get_scaffold(self, smiles: str) -> str:
        """Get Murcko scaffold for a molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return ""

    def split(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data by scaffolds.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Group by scaffold
        scaffold_to_indices = defaultdict(list)
        for idx, smi in enumerate(smiles_list):
            scaffold = self.get_scaffold(smi)
            scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds by size (larger scaffolds first for more stable splits)
        scaffolds = list(scaffold_to_indices.keys())
        scaffolds.sort(key=lambda s: len(scaffold_to_indices[s]), reverse=True)

        # Assign scaffolds to splits
        train_idx, val_idx, test_idx = [], [], []
        train_size = 1 - self.config.val_fraction - self.config.test_fraction
        val_size = self.config.val_fraction
        test_size = self.config.test_fraction

        n_total = len(smiles_list)
        n_train = int(train_size * n_total)
        n_val = int(val_size * n_total)

        np.random.seed(self.config.scaffold_split_seed)
        np.random.shuffle(scaffolds)

        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]

            if len(train_idx) < n_train:
                train_idx.extend(indices)
            elif len(val_idx) < n_val:
                val_idx.extend(indices)
            else:
                test_idx.extend(indices)

        logger.info(f"Scaffold split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return (
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx),
        )


class DataCurationPipeline:
    """
    Complete data curation pipeline.

    Combines ChEMBL fetching, literature curation, and preprocessing.
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or CONFIG.data
        self.fetcher = ChEMBLDataFetcher(self.config)
        self.curator = LiteratureCurator()
        self.filter = MolecularFilter(self.config)
        self.splitter = ScaffoldSplitter(self.config)

    def run(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data curation pipeline.

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        cache_path = self.config.data_dir / "curated_data.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # Fetch and curate data
            df = self._curate_data()

            # Save cache
            df.to_parquet(cache_path)

        # Split data
        splits = self._split_data(df)

        # Save splits
        for split_name, split_df in splits.items():
            split_path = self.config.data_dir / f"{split_name}.parquet"
            split_df.to_parquet(split_path)
            logger.info(f"Saved {split_name} split: {len(split_df)} compounds")

        return splits

    def _curate_data(self) -> pd.DataFrame:
        """Curate and merge all data sources."""
        logger.info("Starting data curation...")

        # 1. Get literature-curated data (highest priority)
        lit_df = self.curator.get_curated_data()

        # 2. Fetch ChEMBL data
        chembl_df = self.fetcher.fetch_all_targets()

        # 3. Merge with literature taking priority
        if len(chembl_df) > 0:
            # Remove ChEMBL entries that conflict with literature
            lit_smiles = set(lit_df['smiles'].unique())
            chembl_df = chembl_df[~chembl_df['smiles'].isin(lit_smiles)]

            # Combine
            df = pd.concat([lit_df, chembl_df], ignore_index=True)
        else:
            df = lit_df

        # 4. Filter molecules
        valid_mask = []
        for smi in df['smiles']:
            passes, _ = self.filter.filter_molecule(smi)
            valid_mask.append(passes)
        df = df[valid_mask].copy()

        # 5. Add stereochemistry info
        stereo_info = df['smiles'].apply(self.filter.get_stereochemistry_info)
        df['num_stereocenters'] = stereo_info.apply(lambda x: x.get('num_stereocenters', 0))
        df['has_stereocenters'] = stereo_info.apply(lambda x: x.get('has_stereocenters', False))
        df['stereo_defined'] = stereo_info.apply(lambda x: x.get('stereo_defined', True))

        # 6. Aggregate by compound (resolve conflicting labels)
        df = self._aggregate_labels(df)

        logger.info(f"Curated {len(df)} compounds")
        return df

    def _aggregate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple measurements per compound into final labels.

        Uses confidence-weighted voting.
        """
        # Group by SMILES and target
        aggregated = []

        for (smiles, target), group in df.groupby(['smiles', 'target']):
            # Weight votes by confidence
            votes = defaultdict(float)
            for _, row in group.iterrows():
                label = row['label']
                conf = row.get('confidence', 0.5)
                if label >= 0:  # Ignore unknown labels
                    votes[label] += conf

            if not votes:
                continue

            # Select highest confidence label
            final_label = max(votes.keys(), key=lambda k: votes[k])
            total_conf = sum(votes.values())
            final_conf = votes[final_label] / total_conf if total_conf > 0 else 0

            aggregated.append({
                'smiles': smiles,
                'target': target,
                'label': final_label,
                'confidence': final_conf,
                'num_measurements': len(group),
                'compound_name': group['compound_name'].iloc[0] if 'compound_name' in group.columns else None,
            })

        return pd.DataFrame(aggregated)

    def _split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data using scaffold splitting."""
        # Get unique SMILES
        unique_smiles = df['smiles'].unique()

        # Get labels for stratification (use DAT as primary)
        dat_labels = []
        for smi in unique_smiles:
            smi_df = df[df['smiles'] == smi]
            dat_df = smi_df[smi_df['target'] == 'DAT']
            if len(dat_df) > 0:
                dat_labels.append(dat_df['label'].iloc[0])
            else:
                dat_labels.append(0)
        dat_labels = np.array(dat_labels)

        # Scaffold split
        train_idx, val_idx, test_idx = self.splitter.split(
            list(unique_smiles),
            dat_labels,
        )

        # Create split DataFrames
        train_smiles = set(unique_smiles[train_idx])
        val_smiles = set(unique_smiles[val_idx])
        test_smiles = set(unique_smiles[test_idx])

        return {
            'train': df[df['smiles'].isin(train_smiles)].copy(),
            'val': df[df['smiles'].isin(val_smiles)].copy(),
            'test': df[df['smiles'].isin(test_smiles)].copy(),
        }

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute dataset statistics."""
        stats = {
            'total_compounds': df['smiles'].nunique(),
            'total_records': len(df),
        }

        for target in ['DAT', 'NET', 'SERT']:
            target_df = df[df['target'] == target]
            stats[f'{target}_total'] = len(target_df)
            stats[f'{target}_substrates'] = len(target_df[target_df['label'] == 2])
            stats[f'{target}_blockers'] = len(target_df[target_df['label'] == 1])
            stats[f'{target}_inactive'] = len(target_df[target_df['label'] == 0])

        # Stereochemistry stats
        stats['compounds_with_stereocenters'] = df[df['has_stereocenters'] == True]['smiles'].nunique()
        stats['compounds_stereo_defined'] = df[df['stereo_defined'] == True]['smiles'].nunique()

        return stats


def main():
    """Run data curation pipeline."""
    print("=" * 60)
    print("StereoGNN Transporter - Data Curation Pipeline")
    print("=" * 60)

    pipeline = DataCurationPipeline()
    splits = pipeline.run(use_cache=False)

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()}:")
        stats = pipeline.get_statistics(split_df)
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
