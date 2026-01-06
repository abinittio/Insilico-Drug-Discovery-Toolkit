"""Check kinetic data quality."""
import pandas as pd

# Load data
train_df = pd.read_parquet('data/kinetic_splits/train.parquet')
val_df = pd.read_parquet('data/kinetic_splits/val.parquet')
test_df = pd.read_parquet('data/kinetic_splits/test.parquet')

print("=" * 60)
print("KINETIC DATA QUALITY CHECK")
print("=" * 60)

print(f"\n=== SPLIT SIZES ===")
print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")
print(f"Test: {len(test_df)}")

print(f"\n=== TARGETS ===")
print(train_df['target'].value_counts())

print(f"\n=== LABELS ===")
print(train_df['label'].value_counts())

print(f"\n=== KINETIC DATA AVAILABILITY ===")
print(f"pKi non-null: {train_df['pKi'].notna().sum()} ({100*train_df['pKi'].notna().mean():.1f}%)")
print(f"pIC50 non-null: {train_df['pIC50'].notna().sum()} ({100*train_df['pIC50'].notna().mean():.1f}%)")
print(f"interaction_mode non-null: {train_df['interaction_mode'].notna().sum()}")
print(f"kinetic_bias non-null: {train_df['kinetic_bias'].notna().sum()}")

print(f"\n=== VALUE RANGES ===")
pki_valid = train_df['pKi'].dropna()
pic50_valid = train_df['pIC50'].dropna()
if len(pki_valid) > 0:
    print(f"pKi: {pki_valid.min():.2f} to {pki_valid.max():.2f} (mean: {pki_valid.mean():.2f})")
if len(pic50_valid) > 0:
    print(f"pIC50: {pic50_valid.min():.2f} to {pic50_valid.max():.2f} (mean: {pic50_valid.mean():.2f})")

print(f"\n=== SAMPLE DATA ===")
print(train_df[['smiles', 'target', 'label', 'pKi', 'pIC50']].head(5).to_string())
