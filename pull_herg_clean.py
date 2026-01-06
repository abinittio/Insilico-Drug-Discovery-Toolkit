"""
Pull CLEAN hERG IC50/Ki data from ChEMBL.
Only quantitative binding data, no % inhibition garbage.
"""
import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
from pathlib import Path

HERG_CHEMBL_ID = 'CHEMBL240'

def pull_clean_herg():
    """Pull only IC50/Ki data with proper values."""
    activity = new_client.activity

    print("Fetching hERG IC50/Ki data from ChEMBL...")

    # Only get IC50, Ki, Kd - quantitative binding data
    results = activity.filter(
        target_chembl_id=HERG_CHEMBL_ID,
        standard_type__in=['IC50', 'Ki', 'Kd'],
        standard_units='nM',
        standard_relation='=',  # Exact values only, no > or <
    ).only([
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_type',
        'standard_value',
        'standard_units',
        'pchembl_value',  # This is the gold - curated pIC50
        'assay_type',
        'assay_description',
    ])

    records = []
    for r in tqdm(results, desc="Fetching"):
        if r.get('canonical_smiles') and r.get('standard_value'):
            records.append({
                'smiles': r['canonical_smiles'],
                'chembl_id': r['molecule_chembl_id'],
                'type': r['standard_type'],
                'value_nM': float(r['standard_value']),
                'pchembl': float(r['pchembl_value']) if r.get('pchembl_value') else None,
                'assay_type': r.get('assay_type', ''),
            })

    df = pd.DataFrame(records)
    print(f"Raw records: {len(df)}")

    # Calculate pIC50 where missing
    df['pIC50'] = df.apply(
        lambda r: r['pchembl'] if pd.notna(r['pchembl']) else -np.log10(r['value_nM'] * 1e-9),
        axis=1
    )

    # Deduplicate: keep highest affinity (lowest IC50 = highest pIC50)
    df = df.sort_values('pIC50', ascending=False).drop_duplicates('smiles', keep='first')

    # Active = IC50 < 10 µM (pIC50 > 5)
    df['active'] = (df['pIC50'] > 5).astype(int)

    # Check for stereochemistry
    df['has_stereo'] = df['smiles'].str.contains(r'@|\\|/', regex=True)

    print(f"\nCleaned dataset:")
    print(f"  Unique molecules: {len(df)}")
    print(f"  With pChEMBL (gold standard): {df['pchembl'].notna().sum()}")
    print(f"  With stereo: {df['has_stereo'].sum()}")
    print(f"  Active (IC50 < 10µM): {(df['active'] == 1).sum()}")
    print(f"  Inactive: {(df['active'] == 0).sum()}")
    print(f"  pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")

    # Save
    out_path = Path('data/toxicity/hERG_clean.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return df

if __name__ == "__main__":
    pull_clean_herg()
