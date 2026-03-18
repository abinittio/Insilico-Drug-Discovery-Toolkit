"""
Verify Data Counts
==================

Quick script to verify we have enough data before training.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data_comprehensive import ComprehensiveLiteratureData
from data_sar_expansion import SARExpander, DecoyGenerator


def main():
    print("=" * 70)
    print("DATA VERIFICATION")
    print("=" * 70)

    # 1. Literature data
    print("\n[1] LITERATURE DATA")
    print("-" * 40)

    lit_df = ComprehensiveLiteratureData.get_all_data()
    print(f"Total records: {len(lit_df)}")
    print(f"Unique compounds: {lit_df['smiles'].nunique()}")

    for target in ['DAT', 'NET', 'SERT']:
        t_df = lit_df[lit_df['target'] == target]
        subs = len(t_df[t_df['label'] == 2])
        block = len(t_df[t_df['label'] == 1])
        inact = len(t_df[t_df['label'] == 0])
        print(f"  {target}: {subs} substrates | {block} blockers | {inact} inactive")

    print("\nBy category:")
    for cat in lit_df['category'].unique():
        count = len(lit_df[lit_df['category'] == cat])
        unique = lit_df[lit_df['category'] == cat]['smiles'].nunique()
        print(f"  {cat}: {unique} compounds ({count} records)")

    # 2. SAR-expanded data
    print("\n[2] SAR-EXPANDED DATA")
    print("-" * 40)

    expander = SARExpander()
    sar_df = expander.generate_all()
    print(f"Total records: {len(sar_df)}")
    print(f"Unique compounds: {sar_df['smiles'].nunique()}")

    for target in ['DAT', 'NET', 'SERT']:
        t_df = sar_df[sar_df['target'] == target]
        subs = len(t_df[t_df['label'] == 2])
        block = len(t_df[t_df['label'] == 1])
        inact = len(t_df[t_df['label'] == 0])
        print(f"  {target}: {subs} substrates | {block} blockers | {inact} inactive")

    # 3. Decoys
    print("\n[3] DECOY DATA")
    print("-" * 40)

    decoy_gen = DecoyGenerator()
    decoy_df = decoy_gen.generate_decoys()
    print(f"Total records: {len(decoy_df)}")
    print(f"Unique compounds: {decoy_df['smiles'].nunique()}")

    # 4. Combined totals
    print("\n" + "=" * 70)
    print("COMBINED TOTALS (before deduplication)")
    print("=" * 70)

    import pandas as pd
    combined = pd.concat([lit_df, sar_df, decoy_df], ignore_index=True)

    print(f"\nTotal records: {len(combined)}")
    print(f"Unique compounds: {combined['smiles'].nunique()}")

    for target in ['DAT', 'NET', 'SERT']:
        t_df = combined[combined['target'] == target]
        subs = t_df[t_df['label'] == 2]['smiles'].nunique()
        block = t_df[t_df['label'] == 1]['smiles'].nunique()
        inact = t_df[t_df['label'] == 0]['smiles'].nunique()
        print(f"  {target}: {subs} substrates | {block} blockers | {inact} inactive")

    # Stereo statistics
    from rdkit import Chem

    stereo_count = 0
    for smi in combined['smiles'].unique():
        mol = Chem.MolFromSmiles(smi)
        if mol:
            chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            if chiral:
                stereo_count += 1

    total_unique = combined['smiles'].nunique()
    print(f"\nCompounds with stereocenters: {stereo_count}/{total_unique} ({100*stereo_count/total_unique:.1f}%)")

    # Check success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    min_substrates = 50
    min_total = 500

    for target in ['DAT', 'NET', 'SERT']:
        t_df = combined[combined['target'] == target]
        subs = t_df[t_df['label'] == 2]['smiles'].nunique()
        status = "PASS" if subs >= min_substrates else "FAIL"
        print(f"  {target} substrates >= {min_substrates}: {subs} [{status}]")

    total = combined['smiles'].nunique()
    status = "PASS" if total >= min_total else "FAIL"
    print(f"  Total compounds >= {min_total}: {total} [{status}]")

    stereo_pct = 100 * stereo_count / total_unique
    status = "PASS" if stereo_pct >= 20 else "WARN"
    print(f"  Stereo compounds >= 20%: {stereo_pct:.1f}% [{status}]")


if __name__ == "__main__":
    main()
