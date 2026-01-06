"""Test kinetic data loading."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

from dataset import KineticTransporterDataset, create_kinetic_dataloaders

print("=" * 60)
print("KINETIC DATA LOADING TEST")
print("=" * 60)

# Test 1: Load from kinetic_splits
print("\n=== Test 1: Loading from kinetic_splits ===")
data_path = Path('data/kinetic_splits')

try:
    dataset = KineticTransporterDataset(
        data_path=data_path,
        split='train',
        pre_featurize=False,  # Don't featurize yet
        use_3d=False,
    )

    print(f"Loaded dataset with {len(dataset)} molecules")
    print(f"DataFrame shape: {dataset.df.shape}")
    print(f"Unique targets in data: {dataset.df['target'].unique().tolist()}")
    print(f"Targets being used (hardcoded): DAT, NET, SERT")

    # Check how many records are DAT/NET/SERT
    monoamine_df = dataset.df[dataset.df['target'].isin(['DAT', 'NET', 'SERT'])]
    print(f"\nRecords with DAT/NET/SERT: {len(monoamine_df)} / {len(dataset.df)} ({100*len(monoamine_df)/len(dataset.df):.1f}%)")

    # Check kinetic labels
    print("\n=== Kinetic Labels Check ===")
    sample_smi = list(dataset.kinetic_labels.keys())[0] if dataset.kinetic_labels else None
    if sample_smi:
        print(f"Sample molecule: {sample_smi[:50]}...")
        print(f"Kinetic labels: {dataset.kinetic_labels[sample_smi]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Try the full dataloader
print("\n=== Test 2: Full Dataloader ===")
try:
    dataloaders = create_kinetic_dataloaders(
        data_path=data_path,
        batch_size=32,
        num_workers=0,
        use_3d=False,
        augment=False,
    )

    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")

    # Get one batch
    batch = next(iter(dataloaders['train']))
    print(f"\nBatch attributes: {batch.keys()}")
    print(f"Batch size: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'unknown'}")

    # Check for kinetic attributes
    for attr in ['pKi_DAT', 'pKi_NET', 'pKi_SERT', 'y_dat', 'y_net', 'y_sert']:
        if hasattr(batch, attr):
            val = getattr(batch, attr)
            print(f"{attr}: shape={val.shape}, non-null={(~val.isnan()).sum().item() if hasattr(val, 'isnan') else 'N/A'}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
