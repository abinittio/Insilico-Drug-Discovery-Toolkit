"""
Data Augmentation for Monoamine Transporter Dataset
====================================================

Since monoamine transporter substrate data is inherently limited
(~500-1000 known substrates worldwide), we use legitimate augmentation:

1. Stereoisomer enumeration (generate all possible stereoisomers)
2. Tautomer enumeration
3. SMILES randomization (for training only)
4. Minor structural perturbations (validated by similarity)

This can expand dataset 3-10x while maintaining chemical validity.
"""

import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StereoisomerEnumerator:
    """
    Enumerate all possible stereoisomers of a molecule.

    Key principle: If we know the activity of a racemic mixture,
    we can infer that at least one stereoisomer is active.
    For defined stereoisomers, we use the known activity.
    """

    def __init__(self):
        self.options = StereoEnumerationOptions(
            tryEmbedding=True,
            onlyUnassigned=False,  # Enumerate all, including assigned
            unique=True,
        )

    def enumerate(self, smiles: str) -> List[str]:
        """
        Enumerate all stereoisomers of a molecule.

        Args:
            smiles: Input SMILES

        Returns:
            List of stereoisomer SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        try:
            isomers = list(EnumerateStereoisomers(mol, options=self.options))
            return [Chem.MolToSmiles(iso, isomericSmiles=True) for iso in isomers]
        except:
            return [smiles]

    def augment_dataset(
        self,
        df: pd.DataFrame,
        max_isomers: int = 8,
    ) -> pd.DataFrame:
        """
        Augment dataset with stereoisomer enumeration.

        For compounds without defined stereochemistry, enumerate
        possible stereoisomers and propagate the label.

        Args:
            df: Input DataFrame with 'smiles', 'target', 'label' columns
            max_isomers: Maximum stereoisomers to enumerate per compound

        Returns:
            Augmented DataFrame
        """
        augmented_records = []

        for _, row in df.iterrows():
            smiles = row['smiles']

            # Check if stereochemistry is defined
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                augmented_records.append(row.to_dict())
                continue

            chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            has_undefined = any(c[1] == '?' for c in chiral_centers)

            if has_undefined or not chiral_centers:
                # Enumerate stereoisomers
                isomers = self.enumerate(smiles)[:max_isomers]

                for iso in isomers:
                    new_row = row.to_dict()
                    new_row['smiles'] = iso
                    new_row['confidence'] = row['confidence'] * 0.8  # Reduce confidence
                    new_row['source'] = f"{row['source']}_stereo_enum"
                    augmented_records.append(new_row)
            else:
                # Keep original (stereochemistry defined)
                augmented_records.append(row.to_dict())

        result_df = pd.DataFrame(augmented_records)

        logger.info(f"Stereoisomer augmentation: {len(df)} -> {len(result_df)} records")
        logger.info(f"  Unique compounds: {df['smiles'].nunique()} -> {result_df['smiles'].nunique()}")

        return result_df


class TautomerEnumerator:
    """
    Enumerate tautomers of molecules.

    Useful for compounds with tautomeric forms that may have
    different activities.
    """

    def __init__(self):
        self.enumerator = rdMolStandardize.TautomerEnumerator()

    def enumerate(self, smiles: str, max_tautomers: int = 5) -> List[str]:
        """Enumerate tautomers."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        try:
            tautomers = self.enumerator.Enumerate(mol)
            result = []
            for t in tautomers:
                if len(result) >= max_tautomers:
                    break
                result.append(Chem.MolToSmiles(t, isomericSmiles=True))
            return result if result else [smiles]
        except:
            return [smiles]


class SMILESRandomizer:
    """
    Generate random SMILES representations.

    Different SMILES orderings can help model learn more robust
    representations (similar to data augmentation in image processing).

    Only used during training, not for augmenting the static dataset.
    """

    def randomize(self, smiles: str, n_variants: int = 10) -> List[str]:
        """Generate random SMILES variants."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]

        variants = set()
        variants.add(Chem.MolToSmiles(mol, isomericSmiles=True))

        for _ in range(n_variants * 2):
            if len(variants) >= n_variants:
                break
            try:
                # Random atom ordering
                random_smiles = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=True,
                    doRandom=True,
                )
                variants.add(random_smiles)
            except:
                pass

        return list(variants)


class SimilarityBasedExpander:
    """
    Expand dataset using similar known active compounds.

    Find compounds in PubChem/ChEMBL that are similar to known actives
    but haven't been tested at these specific targets.

    This is a form of "transfer learning" for small molecule SAR.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    def find_similar(
        self,
        query_smiles: str,
        database_smiles: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar molecules from a database."""
        from rdkit import DataStructs
        from rdkit.Chem import AllChem

        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return []

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

        similarities = []
        for smi in database_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            if sim >= self.threshold:
                similarities.append((smi, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def augment_dataset(
    df: pd.DataFrame,
    use_stereo: bool = True,
    use_tautomers: bool = False,
    max_stereo_isomers: int = 8,
) -> pd.DataFrame:
    """
    Main augmentation function.

    Args:
        df: Input DataFrame
        use_stereo: Enumerate stereoisomers
        use_tautomers: Enumerate tautomers

    Returns:
        Augmented DataFrame
    """
    result = df.copy()

    if use_stereo:
        logger.info("Applying stereoisomer augmentation...")
        enumerator = StereoisomerEnumerator()
        result = enumerator.augment_dataset(result, max_isomers=max_stereo_isomers)

    if use_tautomers:
        logger.info("Applying tautomer augmentation...")
        taut_enum = TautomerEnumerator()
        augmented = []
        for _, row in result.iterrows():
            tautomers = taut_enum.enumerate(row['smiles'], max_tautomers=3)
            for t in tautomers:
                new_row = row.to_dict()
                new_row['smiles'] = t
                augmented.append(new_row)
        result = pd.DataFrame(augmented)

    # Deduplicate
    result = result.drop_duplicates(subset=['smiles', 'target'])

    logger.info(f"Final augmented dataset: {len(result)} records, {result['smiles'].nunique()} compounds")

    return result


# Realistic data size estimation
DATA_SIZE_ESTIMATE = """
REALISTIC DATA SIZE ESTIMATION FOR MONOAMINE TRANSPORTER SUBSTRATES
====================================================================

1. KNOWN SUBSTRATES (Literature)
   - Classic amphetamines & analogs: ~100 compounds
   - Cathinones ("bath salts"): ~50 compounds
   - Tryptamines: ~40 compounds
   - Phenethylamines (non-amphetamine): ~30 compounds
   - Endogenous monoamines & metabolites: ~20 compounds
   - Other (aminoindanes, benzofurans, etc.): ~40 compounds
   SUBTOTAL: ~280 well-characterized substrates

2. KNOWN BLOCKERS (Literature)
   - Cocaine analogs: ~30 compounds
   - SSRIs/SNRIs/NDRIs: ~50 compounds
   - Tricyclic antidepressants: ~30 compounds
   - Other (modafinil, etc.): ~40 compounds
   SUBTOTAL: ~150 well-characterized blockers

3. ChEMBL DATA (with rigorous filtering)
   - DAT: ~2,000-5,000 binding entries (mostly blockers)
   - NET: ~1,500-3,000 binding entries
   - SERT: ~5,000-10,000 binding entries (most studied)
   - After filtering for reliable labels: ~1,000-2,000 compounds
   SUBTOTAL: ~1,000-2,000 (conservative, high-quality)

4. SAR EXPANSION
   - Systematic ring substitutions: ~200 analogs
   - N-substituent variations: ~100 analogs
   - Chain length variations: ~50 analogs
   SUBTOTAL: ~350 SAR-expanded compounds

5. NEGATIVE CONTROLS (Decoys)
   - Druglike compounds with no transporter activity: ~300-500
   SUBTOTAL: ~400 compounds

TOTAL REALISTIC DATASET: ~2,200-3,200 unique compounds

WITH STEREOISOMER AUGMENTATION: ~4,000-6,000 unique compounds

COMPARISON TO BBBP (250k):
- BBBP is a BINARY property (crosses/doesn't cross)
- Transporter substrate is a MECHANISTIC property
- Much less data exists because it requires specific assay types
- Quality > Quantity for this problem

WHAT THIS MEANS:
- The model must learn with less data
- Stereochemistry features are CRITICAL (more information per compound)
- Multi-task learning helps (DAT/NET/SERT share representations)
- The 0.95 AUROC target is achievable but challenging

RECOMMENDED MINIMUM:
- 500 substrates total (across DAT/NET/SERT)
- 300 blockers
- 400 inactive controls
- = ~1,200 unique compounds minimum
- With stereo augmentation: ~2,500+ compounds
"""


if __name__ == "__main__":
    print(DATA_SIZE_ESTIMATE)

    # Test augmentation
    print("\nTesting stereoisomer enumeration...")
    enumerator = StereoisomerEnumerator()

    test_smiles = [
        "CC(N)Cc1ccccc1",  # Racemic amphetamine
        "C[C@H](N)Cc1ccccc1",  # d-Amphetamine (defined)
        "CC(NC)C(=O)c1ccccc1",  # Racemic methcathinone
    ]

    for smi in test_smiles:
        isomers = enumerator.enumerate(smi)
        print(f"  {smi}: {len(isomers)} stereoisomers")
        for iso in isomers[:4]:
            print(f"    - {iso}")
