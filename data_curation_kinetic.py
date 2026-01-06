"""
Kinetic Data Curation Pipeline
==============================

Extends the base data curation to include mechanistic kinetic parameters:
- Binding affinity (Ki/Kd → pKi)
- Functional potency (IC50/EC50 → pIC50)
- Interaction mode classification
- Kinetic bias (uptake vs release preference)

Data Sources:
1. ChEMBL - binding and functional assays with numeric values
2. Literature curation - known mechanism classifications
3. PDSP Ki Database (if available)

Output columns:
- smiles, target, label (original)
- pKi: -log10(Ki in M), NaN if unavailable
- pIC50: -log10(IC50 in M), NaN if unavailable
- interaction_mode: 0=substrate, 1=competitive, 2=non-competitive, 3=partial, -1=unknown
- kinetic_bias: 0-1 (0=pure blocker, 1=pure uptake substrate), NaN if unknown
- confidence: 0-1 label confidence
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logging.warning("ChEMBL client not available.")

from data_curation import (
    DataCurationPipeline,
    LiteratureCurator,
    MolecularFilter,
    ScaffoldSplitter,
    SubstrateBlockerClassifier,
)
from config import CONFIG, DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KineticParameterExtractor:
    """
    Extracts kinetic parameters from assay data.

    Handles unit conversions and standardization:
    - Ki, Kd → pKi (-log10 M)
    - IC50, EC50 → pIC50 (-log10 M)
    """

    # Standard types that represent binding affinity
    BINDING_TYPES = {'Ki', 'Kd', 'KD', 'Kb'}

    # Standard types that represent functional potency
    POTENCY_TYPES = {'IC50', 'EC50', 'AC50'}

    # Unit conversion factors to Molar
    UNIT_TO_MOLAR = {
        'M': 1.0,
        'mM': 1e-3,
        'uM': 1e-6,
        'µM': 1e-6,
        'nM': 1e-9,
        'pM': 1e-12,
        'fM': 1e-15,
        'mol/L': 1.0,
        'mmol/L': 1e-3,
        'umol/L': 1e-6,
        'nmol/L': 1e-9,
        'pmol/L': 1e-12,
    }

    def extract_pki(
        self,
        value: float,
        units: str,
        standard_type: str,
    ) -> Optional[float]:
        """
        Extract pKi from binding assay data.

        Args:
            value: Numeric value
            units: Unit string
            standard_type: Type of measurement (Ki, Kd, etc.)

        Returns:
            pKi value or None if not applicable
        """
        if standard_type not in self.BINDING_TYPES:
            return None

        # Convert value to float if it's a string
        try:
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

        if value is None or value <= 0:
            return None

        # Convert to Molar
        molar_value = self._convert_to_molar(value, units)
        if molar_value is None or molar_value <= 0:
            return None

        # Calculate pKi = -log10(Ki in M)
        pki = -math.log10(molar_value)

        # Sanity check: pKi typically 3-12 for drug-like compounds
        if pki < 2 or pki > 14:
            return None

        return pki

    def extract_pic50(
        self,
        value: float,
        units: str,
        standard_type: str,
    ) -> Optional[float]:
        """
        Extract pIC50 from functional assay data.

        Args:
            value: Numeric value
            units: Unit string
            standard_type: Type of measurement (IC50, EC50, etc.)

        Returns:
            pIC50 value or None if not applicable
        """
        if standard_type not in self.POTENCY_TYPES:
            return None

        # Convert value to float if it's a string
        try:
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

        if value is None or value <= 0:
            return None

        # Convert to Molar
        molar_value = self._convert_to_molar(value, units)
        if molar_value is None or molar_value <= 0:
            return None

        # Calculate pIC50 = -log10(IC50 in M)
        pic50 = -math.log10(molar_value)

        # Sanity check
        if pic50 < 2 or pic50 > 14:
            return None

        return pic50

    def _convert_to_molar(self, value: float, units: str) -> Optional[float]:
        """Convert a value to Molar units."""
        if units is None:
            return None

        factor = self.UNIT_TO_MOLAR.get(units)
        if factor is None:
            # Try common variations
            units_clean = units.strip().replace(' ', '')
            factor = self.UNIT_TO_MOLAR.get(units_clean)

        if factor is None:
            return None

        return value * factor


class InteractionModeClassifier:
    """
    Classifies interaction mode from assay descriptions and data patterns.

    Modes:
    0 = Substrate (transported, may cause release)
    1 = Competitive inhibitor (binds orthosteric site)
    2 = Non-competitive inhibitor (binds allosteric site)
    3 = Partial substrate (partial transport activity)
    """

    # Keywords indicating substrate mechanism
    SUBSTRATE_KEYWORDS = [
        'substrate', 'transported', 'transport substrate',
        'releasing agent', 'releaser', 'efflux',
        'reverse transport', 'carrier-mediated',
        'Km', 'Vmax', 'Michaelis',
    ]

    # Keywords indicating competitive inhibition
    COMPETITIVE_KEYWORDS = [
        'competitive inhibitor', 'competitive antagonist',
        'orthosteric', 'substrate site',
        'competitive binding', 'displaces substrate',
    ]

    # Keywords indicating non-competitive inhibition
    NONCOMPETITIVE_KEYWORDS = [
        'non-competitive', 'noncompetitive', 'non competitive',
        'allosteric', 'uncompetitive',
        'mixed inhibitor', 'mixed-type',
    ]

    # Keywords indicating partial substrate
    PARTIAL_KEYWORDS = [
        'partial substrate', 'partial agonist',
        'partial release', 'weak substrate',
        'partial transport',
    ]

    def classify(
        self,
        assay_description: str,
        activity_comment: str = None,
        standard_type: str = None,
    ) -> Tuple[int, float]:
        """
        Classify interaction mode from assay information.

        Returns:
            Tuple of (mode, confidence)
            mode: 0-3 or -1 for unknown
        """
        text = ' '.join(filter(None, [
            assay_description,
            activity_comment,
        ])).lower()

        # Check for explicit mode keywords
        for kw in self.SUBSTRATE_KEYWORDS:
            if kw.lower() in text:
                return 0, 0.85  # Substrate

        for kw in self.PARTIAL_KEYWORDS:
            if kw.lower() in text:
                return 3, 0.80  # Partial substrate

        for kw in self.NONCOMPETITIVE_KEYWORDS:
            if kw.lower() in text:
                return 2, 0.85  # Non-competitive

        for kw in self.COMPETITIVE_KEYWORDS:
            if kw.lower() in text:
                return 1, 0.85  # Competitive

        # Infer from assay type
        if standard_type in ['Km', 'Vmax']:
            return 0, 0.90  # Kinetic parameters indicate substrate

        if standard_type in ['Ki']:
            # Ki typically from competitive binding
            return 1, 0.60  # Lower confidence

        return -1, 0.0  # Unknown


class KineticBiasCalculator:
    """
    Calculates kinetic bias (uptake vs release preference).

    Kinetic bias = 0: Pure blocker (no transport)
    Kinetic bias = 1: Pure uptake substrate
    Kinetic bias = 0.5: Balanced (releaser or mixed)

    Based on ratio of uptake to release/efflux activity.
    """

    UPTAKE_KEYWORDS = ['uptake', 'transport', 'influx', 'accumulation']
    RELEASE_KEYWORDS = ['release', 'efflux', 'reverse transport', 'outward']

    def calculate_bias(
        self,
        uptake_ic50: Optional[float],
        release_ec50: Optional[float],
        is_substrate: bool,
        is_blocker: bool,
    ) -> Optional[float]:
        """
        Calculate kinetic bias from assay data.

        Args:
            uptake_ic50: IC50 for uptake inhibition (uM)
            release_ec50: EC50 for release induction (uM)
            is_substrate: Whether compound is classified as substrate
            is_blocker: Whether compound is classified as blocker

        Returns:
            Bias value 0-1 or None
        """
        if is_blocker and not is_substrate:
            # Pure blocker
            return 0.0

        if is_substrate:
            if uptake_ic50 is not None and release_ec50 is not None:
                # Calculate relative potency
                # Lower EC50 for release = more releaser-like
                # Higher IC50 for uptake = less uptake inhibitor
                ratio = release_ec50 / (uptake_ic50 + 1e-6)

                # Transform to 0-1 scale
                # ratio < 1: release more potent (releaser)
                # ratio > 1: uptake inhibition more potent
                bias = 1.0 / (1.0 + ratio)  # Sigmoid-like transform
                return np.clip(bias, 0.1, 0.9)
            else:
                # Default for substrates without ratio data
                return 0.7  # Assume mostly uptake-oriented

        return None

    def infer_from_description(self, description: str) -> Optional[float]:
        """Infer kinetic bias from assay description."""
        if not description:
            return None

        desc_lower = description.lower()

        # Count keyword matches
        uptake_score = sum(1 for kw in self.UPTAKE_KEYWORDS if kw in desc_lower)
        release_score = sum(1 for kw in self.RELEASE_KEYWORDS if kw in desc_lower)

        if uptake_score > 0 and release_score == 0:
            return 0.8  # Uptake-oriented
        elif release_score > 0 and uptake_score == 0:
            return 0.3  # Release-oriented
        elif uptake_score > 0 and release_score > 0:
            return 0.5  # Mixed

        return None


class KineticLiteratureCurator(LiteratureCurator):
    """
    Extended literature curator with kinetic parameters.
    """

    # Known compounds with kinetic parameters
    # Format: (SMILES, name, {target: {params}})
    KINETIC_DATA = [
        # Amphetamines - substrates with high uptake bias
        (
            "C[C@H](N)Cc1ccccc1",
            "(+)-Amphetamine",
            {
                "DAT": {"label": 2, "pKi": 7.2, "pIC50": 6.8, "mode": 0, "bias": 0.75},
                "NET": {"label": 2, "pKi": 7.5, "pIC50": 7.0, "mode": 0, "bias": 0.70},
                "SERT": {"label": 1, "pKi": 5.5, "pIC50": 5.0, "mode": 1, "bias": 0.3},
            }
        ),
        (
            "C[C@@H](N)Cc1ccccc1",
            "(-)-Amphetamine",
            {
                "DAT": {"label": 1, "pKi": 5.8, "pIC50": 5.5, "mode": 3, "bias": 0.4},
                "NET": {"label": 1, "pKi": 6.0, "pIC50": 5.7, "mode": 3, "bias": 0.4},
                "SERT": {"label": 0, "pKi": 4.5, "pIC50": 4.2, "mode": -1, "bias": None},
            }
        ),
        (
            "C[C@H](NC)Cc1ccccc1",
            "(+)-Methamphetamine",
            {
                "DAT": {"label": 2, "pKi": 7.5, "pIC50": 7.2, "mode": 0, "bias": 0.70},
                "NET": {"label": 2, "pKi": 7.3, "pIC50": 6.9, "mode": 0, "bias": 0.65},
                "SERT": {"label": 1, "pKi": 5.8, "pIC50": 5.5, "mode": 0, "bias": 0.50},
            }
        ),

        # MDMA - SERT substrate with release
        (
            "C[C@H](NC)Cc1ccc2OCOc2c1",
            "(+)-MDMA",
            {
                "DAT": {"label": 2, "pKi": 6.5, "pIC50": 6.2, "mode": 0, "bias": 0.55},
                "NET": {"label": 2, "pKi": 6.8, "pIC50": 6.5, "mode": 0, "bias": 0.55},
                "SERT": {"label": 2, "pKi": 7.5, "pIC50": 7.2, "mode": 0, "bias": 0.40},  # Strong releaser
            }
        ),

        # Cocaine - pure blocker
        (
            "COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3",
            "Cocaine",
            {
                "DAT": {"label": 1, "pKi": 7.0, "pIC50": 6.5, "mode": 1, "bias": 0.0},
                "NET": {"label": 1, "pKi": 6.5, "pIC50": 6.0, "mode": 1, "bias": 0.0},
                "SERT": {"label": 1, "pKi": 6.8, "pIC50": 6.3, "mode": 1, "bias": 0.0},
            }
        ),

        # SSRIs - competitive SERT blockers
        (
            "CNCCC(Oc1ccc(C(F)(F)F)cc1)c2ccccc2",
            "Fluoxetine",
            {
                "DAT": {"label": 0, "pKi": 5.0, "pIC50": 4.5, "mode": -1, "bias": None},
                "NET": {"label": 0, "pKi": 5.5, "pIC50": 5.0, "mode": -1, "bias": None},
                "SERT": {"label": 1, "pKi": 8.5, "pIC50": 8.0, "mode": 1, "bias": 0.0},
            }
        ),

        # Methylphenidate - DAT/NET blocker
        (
            "COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2",
            "d-threo-Methylphenidate",
            {
                "DAT": {"label": 1, "pKi": 7.8, "pIC50": 7.5, "mode": 1, "bias": 0.1},
                "NET": {"label": 1, "pKi": 7.5, "pIC50": 7.2, "mode": 1, "bias": 0.1},
                "SERT": {"label": 0, "pKi": 4.0, "pIC50": 3.8, "mode": -1, "bias": None},
            }
        ),

        # Cathinones - substrates
        (
            "CC(N)C(=O)c1ccccc1",
            "Cathinone",
            {
                "DAT": {"label": 2, "pKi": 6.8, "pIC50": 6.5, "mode": 0, "bias": 0.65},
                "NET": {"label": 2, "pKi": 7.0, "pIC50": 6.7, "mode": 0, "bias": 0.65},
                "SERT": {"label": 1, "pKi": 5.5, "pIC50": 5.2, "mode": 0, "bias": 0.45},
            }
        ),

        # Endogenous substrates
        (
            "NCCc1ccc(O)c(O)c1",
            "Dopamine",
            {
                "DAT": {"label": 2, "pKi": 6.0, "pIC50": 5.5, "mode": 0, "bias": 0.90},
                "NET": {"label": 2, "pKi": 6.2, "pIC50": 5.8, "mode": 0, "bias": 0.85},
                "SERT": {"label": 0, "pKi": 4.0, "pIC50": 3.5, "mode": -1, "bias": None},
            }
        ),
        (
            "NCCc1c[nH]c2ccc(O)cc12",
            "Serotonin",
            {
                "DAT": {"label": 0, "pKi": 4.5, "pIC50": 4.0, "mode": -1, "bias": None},
                "NET": {"label": 0, "pKi": 5.0, "pIC50": 4.5, "mode": -1, "bias": None},
                "SERT": {"label": 2, "pKi": 7.0, "pIC50": 6.5, "mode": 0, "bias": 0.85},
            }
        ),

        # GBR-12909 - selective DAT blocker
        (
            "Fc1ccc(C(c2ccc(F)cc2)N3CCCC3)cc1",
            "GBR-12909",
            {
                "DAT": {"label": 1, "pKi": 9.0, "pIC50": 8.5, "mode": 1, "bias": 0.0},
                "NET": {"label": 0, "pKi": 5.5, "pIC50": 5.0, "mode": -1, "bias": None},
                "SERT": {"label": 0, "pKi": 5.0, "pIC50": 4.5, "mode": -1, "bias": None},
            }
        ),
    ]

    def get_kinetic_curated_data(self) -> pd.DataFrame:
        """Generate curated dataset with kinetic parameters."""
        records = []

        for smiles, name, target_data in self.KINETIC_DATA:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            for target, params in target_data.items():
                records.append({
                    'smiles': canonical,
                    'compound_name': name,
                    'target': target,
                    'label': params['label'],
                    'pKi': params.get('pKi', float('nan')),
                    'pIC50': params.get('pIC50', float('nan')),
                    'interaction_mode': params.get('mode', -1),
                    'kinetic_bias': params.get('bias') if params.get('bias') is not None else float('nan'),
                    'confidence': 1.0,
                    'source': 'literature_kinetic',
                })

        return pd.DataFrame(records)


class ChEMBLKineticFetcher:
    """
    Fetches kinetic parameters from ChEMBL with retry logic and checkpointing.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.classifier = SubstrateBlockerClassifier()
        self.kinetic_extractor = KineticParameterExtractor()
        self.mode_classifier = InteractionModeClassifier()
        self.bias_calculator = KineticBiasCalculator()
        self.cache_dir = config.data_dir / "chembl_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if CHEMBL_AVAILABLE:
            self.activity = new_client.activity
            self.assay = new_client.assay
        else:
            self.activity = None
            self.assay = None

    def _get_checkpoint_path(self, target_name: str) -> Path:
        """Get checkpoint file path for a target."""
        return self.cache_dir / f"checkpoint_{target_name}_kinetic.pkl"

    def _save_checkpoint(self, target_name: str, records: list, last_idx: int):
        """Save progress checkpoint."""
        import pickle
        checkpoint = {'records': records, 'last_idx': last_idx}
        with open(self._get_checkpoint_path(target_name), 'wb') as f:
            pickle.dump(checkpoint, f)

    def _load_checkpoint(self, target_name: str) -> tuple:
        """Load checkpoint if exists. Returns (records, start_idx)."""
        import pickle
        cp_path = self._get_checkpoint_path(target_name)
        if cp_path.exists():
            with open(cp_path, 'rb') as f:
                checkpoint = pickle.load(f)
                logger.info(f"Resuming {target_name} from checkpoint idx {checkpoint['last_idx']}")
                return checkpoint['records'], checkpoint['last_idx'] + 1
        return [], 0

    def _clear_checkpoint(self, target_name: str):
        """Clear checkpoint after successful completion."""
        cp_path = self._get_checkpoint_path(target_name)
        if cp_path.exists():
            cp_path.unlink()

    def _fetch_with_retry(self, fetch_fn, max_retries: int = 3):
        """Execute fetch function with retry logic."""
        import time
        for attempt in range(max_retries):
            try:
                return fetch_fn()
            except Exception as e:
                wait_time = (attempt + 1) * 10
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        return []

    def fetch_kinetic_data(self, target_name: str, target_chembl_id: str,
                            max_records: int = 15000, batch_size: int = 500) -> pd.DataFrame:
        """Fetch activity data with kinetic parameters for a target with checkpointing."""
        logger.info(f"Fetching kinetic data for {target_name} (max: {max_records})")

        if not CHEMBL_AVAILABLE:
            return pd.DataFrame()

        # Load checkpoint if exists
        records, start_offset = self._load_checkpoint(target_name)
        if records:
            logger.info(f"Resuming from {len(records)} cached records, offset {start_offset}")

        # Fetch in batches with progress
        query = self.activity.filter(
            target_chembl_id=target_chembl_id,
            assay_type__in=['B', 'F'],
        ).only([
            'molecule_chembl_id',
            'canonical_smiles',
            'standard_value',
            'standard_units',
            'standard_type',
            'pchembl_value',
            'assay_type',
            'assay_description',
            'activity_comment',
        ])

        offset = start_offset
        activities = []

        logger.info(f"Fetching in batches of {batch_size}...")
        while len(activities) < max_records:
            try:
                batch = list(query[offset:offset + batch_size])
                if not batch:
                    break
                activities.extend(batch)
                offset += batch_size
                logger.info(f"  {target_name}: {len(activities)} records fetched...")

                # Save progress every 2000 records
                if len(activities) % 2000 == 0:
                    self._save_checkpoint(target_name, records, offset)

            except Exception as e:
                logger.warning(f"Batch fetch failed at offset {offset}: {e}")
                import time
                time.sleep(10)
                continue

        if not activities and not records:
            logger.warning(f"No activities fetched for {target_name}")
            return pd.DataFrame()

        logger.info(f"Fetched {len(activities)} activities for {target_name}")

        for i, act in enumerate(tqdm(activities, desc=f"Processing {target_name}")):
            smiles = act.get('canonical_smiles')
            if not smiles:
                continue

            value = act.get('standard_value')
            units = act.get('standard_units')
            std_type = act.get('standard_type')
            assay_desc = act.get('assay_description', '')
            comment = act.get('activity_comment', '')

            # Extract kinetic parameters
            pki = self.kinetic_extractor.extract_pki(value, units, std_type)
            pic50 = self.kinetic_extractor.extract_pic50(value, units, std_type)

            # Use pchembl if available and we couldn't extract
            pchembl = act.get('pchembl_value')
            if pchembl and pki is None and std_type in ['Ki', 'Kd']:
                pki = pchembl
            if pchembl and pic50 is None and std_type in ['IC50', 'EC50']:
                pic50 = pchembl

            # Classify interaction mode
            mode, mode_conf = self.mode_classifier.classify(assay_desc, comment, std_type)

            # Get activity label
            label, label_conf = self.classifier.classify_from_assay(
                act.get('assay_type', ''),
                assay_desc,
                value,
                std_type,
                units,
            )

            # Infer kinetic bias
            bias = self.bias_calculator.infer_from_description(assay_desc)

            records.append({
                'molecule_chembl_id': act.get('molecule_chembl_id'),
                'smiles': smiles,
                'target': target_name,
                'label': label,
                'pKi': pki if pki else float('nan'),
                'pIC50': pic50 if pic50 else float('nan'),
                'interaction_mode': mode,
                'kinetic_bias': bias if bias is not None else float('nan'),
                'confidence': max(label_conf, mode_conf) if label >= 0 else mode_conf,
                'assay_description': assay_desc,
                'standard_type': std_type,
            })

            # Checkpoint every 500 records during processing
            if (i + 1) % 500 == 0:
                self._save_checkpoint(target_name, records, offset)
                logger.info(f"  Processed {i + 1}/{len(activities)}")

        # Clear checkpoint on successful completion
        self._clear_checkpoint(target_name)
        logger.info(f"Completed {target_name}: {len(records)} records")

        return pd.DataFrame(records)

    def fetch_all_kinetic(self) -> pd.DataFrame:
        """Fetch kinetic data for all targets."""
        all_dfs = []

        for target_name, chembl_id in self.config.chembl_targets.items():
            df = self.fetch_kinetic_data(target_name, chembl_id)
            if len(df) > 0:
                all_dfs.append(df)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def fetch_exhaustive_kinetic(self) -> pd.DataFrame:
        """
        EXHAUSTIVE data fetching - gets ALL available monoamine transporter data.
        Includes: primary targets + related targets + all assay types + keyword searches.
        """
        all_dfs = []

        # Extended target list for maximum coverage
        extended_targets = {
            # Primary monoamine transporters
            "DAT": "CHEMBL238",   # SLC6A3 - Dopamine transporter
            "NET": "CHEMBL222",   # SLC6A2 - Norepinephrine transporter
            "SERT": "CHEMBL228",  # SLC6A4 - Serotonin transporter
            # Vesicular transporters (related)
            "VMAT1": "CHEMBL1907601",  # SLC18A1
            "VMAT2": "CHEMBL1966",     # SLC18A2 - Key for amphetamine action
            # TAAR1 removed - insufficient data (only 1 record in ChEMBL)
            # Dopamine receptors (for selectivity data)
            "D1": "CHEMBL2056",
            "D2": "CHEMBL217",
            "D3": "CHEMBL234",
            "D4": "CHEMBL219",
            "D5": "CHEMBL1850",
            # Adrenergic receptors
            "Alpha1A": "CHEMBL229",
            "Alpha2A": "CHEMBL1867",
            "Beta1": "CHEMBL213",
            # Serotonin receptors
            "5HT1A": "CHEMBL214",
            "5HT2A": "CHEMBL224",
            "5HT2B": "CHEMBL1833",
            "5HT2C": "CHEMBL225",
        }

        logger.info("="*60)
        logger.info("EXHAUSTIVE ChEMBL Data Fetching")
        logger.info(f"Targets: {len(extended_targets)}")
        logger.info("="*60)

        # Higher limits for primary targets, lower for secondary
        primary_targets = {'DAT', 'NET', 'SERT', 'VMAT2'}

        for target_name, chembl_id in extended_targets.items():
            max_recs = 50000 if target_name in primary_targets else 10000
            logger.info(f"\n[{target_name}] Fetching from {chembl_id} (max: {max_recs})...")
            df = self.fetch_kinetic_data(target_name, chembl_id, max_records=max_recs)
            if len(df) > 0:
                all_dfs.append(df)
                logger.info(f"  -> {len(df)} records")

        # Also search by assay keywords for amphetamine-specific assays
        amphetamine_keywords = [
            "amphetamine", "methamphetamine", "cathinone",
            "phenethylamine", "dopamine release", "dopamine uptake",
            "norepinephrine release", "serotonin release",
            "stimulant", "psychostimulant", "substrate release",
            "transporter substrate", "DAT substrate", "NET substrate",
            "SERT substrate", "monoamine release", "efflux",
        ]

        logger.info("\nSearching by amphetamine-related keywords...")
        keyword_records = self._search_by_keywords(amphetamine_keywords)
        if keyword_records:
            all_dfs.append(pd.DataFrame(keyword_records))
            logger.info(f"  -> {len(keyword_records)} keyword-matched records")

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            # Deduplicate by SMILES + target
            combined = combined.drop_duplicates(subset=['smiles', 'target'], keep='first')
            logger.info(f"\nTotal unique records: {len(combined)}")
            return combined
        return pd.DataFrame()

    def _search_by_keywords(self, keywords: List[str]) -> List[dict]:
        """Search ChEMBL assays by keywords."""
        if not CHEMBL_AVAILABLE:
            return []

        records = []
        for keyword in keywords:
            try:
                logger.info(f"  Searching: {keyword}")
                # Search assays by description
                assays = list(self.assay.filter(
                    description__icontains=keyword
                ).only(['assay_chembl_id', 'assay_type', 'description']))

                for assay in assays[:50]:  # Limit per keyword
                    assay_id = assay.get('assay_chembl_id')
                    if not assay_id:
                        continue

                    # Get activities for this assay
                    activities = self._fetch_with_retry(
                        lambda aid=assay_id: list(self.activity.filter(
                            assay_chembl_id=aid
                        ).only([
                            'molecule_chembl_id', 'canonical_smiles',
                            'standard_value', 'standard_units', 'standard_type',
                            'pchembl_value', 'assay_description',
                        ])[:100])  # Limit per assay
                    )

                    for act in activities:
                        smiles = act.get('canonical_smiles')
                        if not smiles:
                            continue

                        value = act.get('standard_value')
                        units = act.get('standard_units')
                        std_type = act.get('standard_type')
                        assay_desc = act.get('assay_description', '')

                        pki = self.kinetic_extractor.extract_pki(value, units, std_type)
                        pic50 = self.kinetic_extractor.extract_pic50(value, units, std_type)

                        # Infer target from description
                        target = self._infer_target_from_description(assay_desc)

                        records.append({
                            'molecule_chembl_id': act.get('molecule_chembl_id'),
                            'smiles': smiles,
                            'target': target,
                            'label': -1,  # Unknown, needs classification
                            'pKi': pki if pki else float('nan'),
                            'pIC50': pic50 if pic50 else float('nan'),
                            'interaction_mode': -1,
                            'kinetic_bias': float('nan'),
                            'confidence': 0.5,
                            'assay_description': assay_desc[:500],
                            'standard_type': std_type,
                            'source': f'keyword:{keyword}',
                        })

            except Exception as e:
                logger.warning(f"Keyword search failed for '{keyword}': {e}")
                continue

        return records

    def _infer_target_from_description(self, desc: str) -> str:
        """Infer target from assay description."""
        desc_lower = desc.lower()
        if 'dopamine' in desc_lower or 'dat' in desc_lower:
            return 'DAT'
        elif 'norepinephrine' in desc_lower or 'noradrenaline' in desc_lower or 'net' in desc_lower:
            return 'NET'
        elif 'serotonin' in desc_lower or '5-ht' in desc_lower or 'sert' in desc_lower:
            return 'SERT'
        elif 'vmat' in desc_lower:
            return 'VMAT2'
        # TAAR1 removed - insufficient data
        return 'UNKNOWN'


class KineticDataCurationPipeline(DataCurationPipeline):
    """
    Extended data curation pipeline with kinetic parameters.
    """

    def __init__(self, config: DataConfig = None):
        super().__init__(config)
        self.kinetic_curator = KineticLiteratureCurator()
        self.kinetic_fetcher = ChEMBLKineticFetcher(self.config)

    def run(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """Run kinetic data curation pipeline."""
        cache_path = self.config.data_dir / "kinetic_curated_data.parquet"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached kinetic data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            df = self._curate_kinetic_data()
            df.to_parquet(cache_path)

        # Split data
        splits = self._split_data(df)

        # Save splits
        for split_name, split_df in splits.items():
            split_path = self.config.data_dir / f"{split_name}.parquet"
            split_df.to_parquet(split_path)
            logger.info(f"Saved {split_name} split: {len(split_df)} records")

        return splits

    def _curate_kinetic_data(self) -> pd.DataFrame:
        """Curate data with kinetic parameters."""
        logger.info("Starting kinetic data curation...")

        # 1. Get literature kinetic data (highest priority)
        lit_df = self.kinetic_curator.get_kinetic_curated_data()
        logger.info(f"Literature kinetic records: {len(lit_df)}")

        # 2. Get base literature data
        base_lit_df = self.curator.get_curated_data()
        # Add empty kinetic columns
        for col in ['pKi', 'pIC50', 'interaction_mode', 'kinetic_bias']:
            if col not in base_lit_df.columns:
                base_lit_df[col] = float('nan') if col != 'interaction_mode' else -1

        # Merge, preferring kinetic data
        lit_kinetic_smiles = set(lit_df['smiles'].unique())
        base_lit_df = base_lit_df[~base_lit_df['smiles'].isin(lit_kinetic_smiles)]

        # 3. Fetch ChEMBL kinetic data
        chembl_df = self.kinetic_fetcher.fetch_all_kinetic()
        logger.info(f"ChEMBL kinetic records: {len(chembl_df)}")

        # 4. Combine all sources
        all_lit_smiles = set(lit_df['smiles'].unique()) | set(base_lit_df['smiles'].unique())
        if len(chembl_df) > 0:
            chembl_df = chembl_df[~chembl_df['smiles'].isin(all_lit_smiles)]

        df = pd.concat([lit_df, base_lit_df, chembl_df], ignore_index=True)

        # 5. Filter molecules
        valid_mask = []
        for smi in df['smiles']:
            passes, _ = self.filter.filter_molecule(smi)
            valid_mask.append(passes)
        df = df[valid_mask].copy()

        # 6. Add stereochemistry info
        stereo_info = df['smiles'].apply(self.filter.get_stereochemistry_info)
        df['num_stereocenters'] = stereo_info.apply(lambda x: x.get('num_stereocenters', 0))
        df['has_stereocenters'] = stereo_info.apply(lambda x: x.get('has_stereocenters', False))

        # 7. Aggregate by compound
        df = self._aggregate_kinetic_labels(df)

        logger.info(f"Curated {len(df)} kinetic records")
        return df

    def _aggregate_kinetic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate kinetic labels with proper averaging."""
        aggregated = []

        for (smiles, target), group in df.groupby(['smiles', 'target']):
            # Activity label - confidence weighted
            votes = defaultdict(float)
            for _, row in group.iterrows():
                label = row['label']
                conf = row.get('confidence', 0.5)
                if label >= 0:
                    votes[label] += conf

            if not votes:
                continue

            final_label = max(votes.keys(), key=lambda k: votes[k])
            total_conf = sum(votes.values())
            final_conf = votes[final_label] / total_conf if total_conf > 0 else 0

            # Kinetic parameters - take mean of valid values
            valid_pki = group['pKi'].dropna()
            valid_pic50 = group['pIC50'].dropna()
            valid_bias = group['kinetic_bias'].dropna()

            # Mode - majority vote among valid
            valid_modes = group[group['interaction_mode'] >= 0]['interaction_mode']
            if len(valid_modes) > 0:
                final_mode = valid_modes.mode().iloc[0] if len(valid_modes.mode()) > 0 else -1
            else:
                final_mode = -1

            aggregated.append({
                'smiles': smiles,
                'target': target,
                'label': final_label,
                'pKi': valid_pki.mean() if len(valid_pki) > 0 else float('nan'),
                'pIC50': valid_pic50.mean() if len(valid_pic50) > 0 else float('nan'),
                'interaction_mode': int(final_mode),
                'kinetic_bias': valid_bias.mean() if len(valid_bias) > 0 else float('nan'),
                'confidence': final_conf,
                'num_measurements': len(group),
                'compound_name': group['compound_name'].iloc[0] if 'compound_name' in group.columns and pd.notna(group['compound_name'].iloc[0]) else None,
            })

        return pd.DataFrame(aggregated)

    def get_kinetic_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute kinetic-specific statistics."""
        # Basic stats (without requiring stereo columns from parent)
        stats = {
            'total_records': len(df),
            'total_compounds': df['smiles'].nunique() if 'smiles' in df.columns else 0,
        }

        # Per-target basic stats
        for target in ['DAT', 'NET', 'SERT']:
            target_df = df[df['target'] == target]
            stats[f'{target}_total'] = len(target_df)
            if 'label' in df.columns:
                stats[f'{target}_substrates'] = len(target_df[target_df['label'] == 2])
                stats[f'{target}_blockers'] = len(target_df[target_df['label'] == 1])
                stats[f'{target}_inactive'] = len(target_df[target_df['label'] == 0])

        # Kinetic-specific stats
        for target in ['DAT', 'NET', 'SERT']:
            target_df = df[df['target'] == target]

            # Kinetic coverage
            stats[f'{target}_pKi_available'] = target_df['pKi'].notna().sum()
            stats[f'{target}_pIC50_available'] = target_df['pIC50'].notna().sum()
            stats[f'{target}_mode_available'] = (target_df['interaction_mode'] >= 0).sum()
            stats[f'{target}_bias_available'] = target_df['kinetic_bias'].notna().sum()

            # Mode distribution
            for mode, name in [(0, 'substrate'), (1, 'competitive'),
                               (2, 'noncompetitive'), (3, 'partial')]:
                stats[f'{target}_mode_{name}'] = (target_df['interaction_mode'] == mode).sum()

            # Value ranges
            if target_df['pKi'].notna().sum() > 0:
                stats[f'{target}_pKi_mean'] = target_df['pKi'].mean()
                stats[f'{target}_pKi_std'] = target_df['pKi'].std()
            if target_df['pIC50'].notna().sum() > 0:
                stats[f'{target}_pIC50_mean'] = target_df['pIC50'].mean()
                stats[f'{target}_pIC50_std'] = target_df['pIC50'].std()

        return stats


def main():
    """Run kinetic data curation pipeline."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exhaustive', action='store_true',
                        help='Run exhaustive data fetching (all targets + keywords)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-fetch even if cache exists')
    args = parser.parse_args()

    print("=" * 60)
    print("StereoGNN Kinetic Data Curation Pipeline")
    print("=" * 60)

    if args.exhaustive:
        print("\n*** EXHAUSTIVE MODE: Fetching ALL available data ***\n")
        from config import DataConfig
        config = DataConfig()
        fetcher = ChEMBLKineticFetcher(config)
        df = fetcher.fetch_exhaustive_kinetic()

        # Save raw exhaustive data
        output_path = config.data_dir / "exhaustive_kinetic_data.parquet"
        df.to_parquet(output_path)
        print(f"\nSaved exhaustive data to: {output_path}")
        print(f"Total records: {len(df)}")
        print(f"Unique molecules: {df['smiles'].nunique()}")
        print("\nTarget distribution:")
        print(df['target'].value_counts())
        return

    pipeline = KineticDataCurationPipeline()
    splits = pipeline.run(use_cache=not args.no_cache)

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()}:")
        stats = pipeline.get_kinetic_statistics(split_df)

        # Basic stats
        print(f"  Total compounds: {stats.get('total_compounds', 0)}")
        print(f"  Total records: {stats.get('total_records', 0)}")

        # Per-target stats
        for target in ['DAT', 'NET', 'SERT']:
            print(f"\n  {target}:")
            print(f"    Total: {stats.get(f'{target}_total', 0)}")
            print(f"    Substrates: {stats.get(f'{target}_substrates', 0)}")
            print(f"    Blockers: {stats.get(f'{target}_blockers', 0)}")
            print(f"    pKi available: {stats.get(f'{target}_pKi_available', 0)}")
            print(f"    pIC50 available: {stats.get(f'{target}_pIC50_available', 0)}")
            print(f"    Mode labeled: {stats.get(f'{target}_mode_available', 0)}")


if __name__ == "__main__":
    main()
