"""
Prepare exhaustive kinetic data for training.

1. Enumerates stereoisomers for all molecules
2. Splits into train/val/test using scaffold-based splitting

Usage:
    python prepare_kinetic_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from config import CONFIG


def enumerate_stereoisomers(smiles: str, max_isomers: int = 8) -> list:
    """
    Enumerate all stereoisomers of a molecule.

    Args:
        smiles: Input SMILES
        max_isomers: Maximum number of isomers to return

    Returns:
        List of stereoisomer SMILES (includes original if valid)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  # Return original if can't parse

    try:
        opts = StereoEnumerationOptions(
            tryEmbedding=False,
            unique=True,
            maxIsomers=max_isomers
        )
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if isomers:
            return [Chem.MolToSmiles(iso) for iso in isomers]
        else:
            return [Chem.MolToSmiles(mol)]
    except Exception:
        return [smiles]


def expand_stereoisomers(df: pd.DataFrame, max_isomers: int = 8) -> pd.DataFrame:
    """
    Expand dataframe by enumerating stereoisomers for each molecule.

    Each row is duplicated for each stereoisomer, preserving all other columns.
    """
    print(f"\nEnumerating stereoisomers (max {max_isomers} per molecule)...")

    expanded_rows = []
    unique_smiles = df['smiles'].unique()

    # Cache stereoisomer mappings
    smiles_to_isomers = {}
    for smi in tqdm(unique_smiles, desc="Enumerating stereoisomers"):
        smiles_to_isomers[smi] = enumerate_stereoisomers(smi, max_isomers)

    # Expand dataframe
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding rows"):
        original_smiles = row['smiles']
        isomers = smiles_to_isomers.get(original_smiles, [original_smiles])

        for iso_smiles in isomers:
            new_row = row.copy()
            new_row['smiles'] = iso_smiles
            new_row['original_smiles'] = original_smiles
            expanded_rows.append(new_row)

    result_df = pd.DataFrame(expanded_rows)

    print(f"Stereoisomer expansion: {len(df)} -> {len(result_df)} records")
    print(f"Unique molecules: {df['smiles'].nunique()} -> {result_df['smiles'].nunique()}")

    return result_df


def get_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def scaffold_split(df: pd.DataFrame,
                   val_frac: float = 0.15,
                   test_frac: float = 0.15,
                   seed: int = 42) -> tuple:
    """
    Split dataframe by scaffolds.

    Uses 'original_smiles' if available (for stereoisomer-expanded data)
    to keep all isomers of a molecule in the same split.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Use original_smiles for scaffolding if available (keeps stereoisomers together)
    scaffold_col = 'original_smiles' if 'original_smiles' in df.columns else 'smiles'
    unique_smiles = df[scaffold_col].unique()
    smiles_to_scaffold = {smi: get_scaffold(smi) for smi in unique_smiles}

    # Group SMILES by scaffold
    scaffold_to_smiles = defaultdict(list)
    for smi, scaffold in smiles_to_scaffold.items():
        scaffold_to_smiles[scaffold].append(smi)

    # Sort scaffolds by size (larger first for stability)
    scaffolds = list(scaffold_to_smiles.keys())
    scaffolds.sort(key=lambda s: len(scaffold_to_smiles[s]), reverse=True)

    # Shuffle scaffolds
    np.random.seed(seed)
    np.random.shuffle(scaffolds)

    # Calculate split sizes
    n_total = len(unique_smiles)
    n_train = int((1 - val_frac - test_frac) * n_total)
    n_val = int(val_frac * n_total)

    # Assign scaffolds to splits
    train_smiles, val_smiles, test_smiles = set(), set(), set()

    for scaffold in scaffolds:
        smiles_in_scaffold = scaffold_to_smiles[scaffold]

        if len(train_smiles) < n_train:
            train_smiles.update(smiles_in_scaffold)
        elif len(val_smiles) < n_val:
            val_smiles.update(smiles_in_scaffold)
        else:
            test_smiles.update(smiles_in_scaffold)

    # Split dataframe (use scaffold_col for membership check)
    train_df = df[df[scaffold_col].isin(train_smiles)].copy()
    val_df = df[df[scaffold_col].isin(val_smiles)].copy()
    test_df = df[df[scaffold_col].isin(test_smiles)].copy()

    return train_df, val_df, test_df


def main():
    data_dir = CONFIG.data.data_dir
    input_file = data_dir / "exhaustive_kinetic_data.parquet"

    print("=" * 60)
    print("Preparing Kinetic Data for Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Total records: {len(df)}")
    print(f"Unique molecules: {df['smiles'].nunique()}")

    # Filter to core transporters only (remove TAAR1, keep receptors for context)
    core_targets = {'DAT', 'NET', 'SERT', 'VMAT1', 'VMAT2'}
    receptor_targets = {'D1', 'D2', 'D3', 'D4', 'D5',
                        '5HT1A', '5HT2A', '5HT2B', '5HT2C',
                        'Alpha1A', 'Alpha2A', 'Beta1'}
    valid_targets = core_targets | receptor_targets

    # Keep UNKNOWN for now (will be filtered during training if needed)
    df_filtered = df[df['target'].isin(valid_targets) | (df['target'] == 'UNKNOWN')]
    print(f"\nAfter filtering targets: {len(df_filtered)} records")
    print(f"Target distribution:\n{df_filtered['target'].value_counts()}")

    # Expand stereoisomers
    df_expanded = expand_stereoisomers(df_filtered, max_isomers=8)

    # Scaffold split (use original_smiles for scaffold to keep isomers together)
    print("\nPerforming scaffold-based split...")
    train_df, val_df, test_df = scaffold_split(
        df_expanded,
        val_frac=CONFIG.data.val_fraction,
        test_frac=CONFIG.data.test_fraction,
        seed=CONFIG.data.scaffold_split_seed
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} records ({train_df['smiles'].nunique()} molecules)")
    print(f"  Val:   {len(val_df)} records ({val_df['smiles'].nunique()} molecules)")
    print(f"  Test:  {len(test_df)} records ({test_df['smiles'].nunique()} molecules)")

    # Save splits
    output_dir = data_dir / "kinetic_splits"
    output_dir.mkdir(exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"\nSaved splits to: {output_dir}")
    print("  - train.parquet")
    print("  - val.parquet")
    print("  - test.parquet")

    print("\n" + "=" * 60)
    print("Ready for training!")
    print(f"Run: python run_training_kinetic.py --data-dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
