"""
Kinetic Model Training v3 - Full Dataset + DAT Upweighting
===========================================================

Trains on 63k molecules with:
- New ChEMBL data merged in
- DAT upweighted in loss (2x)
- Full stereo-awareness

Usage:
    python train_kinetic_v3.py --epochs 100
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from model import StereoGNNKinetic, count_parameters
from featurizer import MoleculeGraphFeaturizer
from config import CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DAT upweighting factor
DAT_WEIGHT = 2.0


def merge_new_data(data_dir: Path) -> pd.DataFrame:
    """Merge existing data with new ChEMBL data."""
    print("Loading existing data...")
    train = pd.read_parquet(data_dir / 'train.parquet')
    val = pd.read_parquet(data_dir / 'val.parquet')
    test = pd.read_parquet(data_dir / 'test.parquet')

    print(f"  Existing: train={len(train)}, val={len(val)}, test={len(test)}")

    # Load new ChEMBL data
    new_data_path = data_dir.parent / 'new_chembl_all_transporters.csv'
    if new_data_path.exists():
        print(f"\nLoading new ChEMBL data from {new_data_path}...")
        new_df = pd.read_csv(new_data_path)

        # Format to match existing schema
        new_records = []
        for _, row in new_df.iterrows():
            new_records.append({
                'smiles': row['smiles'],
                'target': row['target'],
                'label': -1,  # Unknown substrate/blocker status
                'pKi': row['pValue'] if row['assay_type'] == 'Ki' else None,
                'pIC50': row['pValue'] if row['assay_type'] == 'IC50' else None,
                'interaction_mode': None,
                'kinetic_bias': None,
                'source': 'chembl_new',
            })

        new_formatted = pd.DataFrame(new_records)
        print(f"  New data: {len(new_formatted)} records, {new_formatted['smiles'].nunique()} unique")
        print(f"  By target: {new_formatted['target'].value_counts().to_dict()}")

        # Add to training set
        train = pd.concat([train, new_formatted], ignore_index=True)
        print(f"  Combined train: {len(train)} records")
    else:
        print(f"  No new data file found at {new_data_path}")

    return train, val, test


def load_kinetic_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Filter for monoamine transporters."""
    df = df[df['target'].isin(['DAT', 'NET', 'SERT'])].copy()
    print(f"  {name}: {len(df)} records, {df['smiles'].nunique()} unique molecules")
    print(f"    Targets: {df['target'].value_counts().to_dict()}")
    return df


def create_graph_data(
    smiles: str,
    labels: Dict[str, int],
    kinetics: Dict[str, Dict],
    featurizer: MoleculeGraphFeaturizer,
) -> Optional[Data]:
    """Create PyG Data object with kinetic labels."""
    try:
        data = featurizer.featurize(smiles, labels)
        if data is None:
            return None

        for target in ['DAT', 'NET', 'SERT']:
            k = kinetics.get(target, {})

            pki = k.get('pKi', float('nan'))
            setattr(data, f'pKi_{target}', torch.tensor([pki], dtype=torch.float32))

            pic50 = k.get('pIC50', float('nan'))
            setattr(data, f'pIC50_{target}', torch.tensor([pic50], dtype=torch.float32))

            mode = k.get('interaction_mode', -1)
            if pd.isna(mode):
                mode = -1
            setattr(data, f'mode_{target}', torch.tensor([int(mode)], dtype=torch.long))

            bias = k.get('kinetic_bias', float('nan'))
            setattr(data, f'bias_{target}', torch.tensor([bias], dtype=torch.float32))

        return data
    except Exception:
        return None


def prepare_dataset(
    df: pd.DataFrame,
    featurizer: MoleculeGraphFeaturizer,
    desc: str = "Processing",
) -> List[Data]:
    """Convert DataFrame to list of PyG Data objects."""
    graphs = []
    molecules = df['smiles'].unique()

    for smi in tqdm(molecules, desc=desc):
        mol_df = df[df['smiles'] == smi]

        labels = {'DAT': -1, 'NET': -1, 'SERT': -1}
        kinetics = {'DAT': {}, 'NET': {}, 'SERT': {}}

        for target in ['DAT', 'NET', 'SERT']:
            target_df = mol_df[mol_df['target'] == target]
            if len(target_df) > 0:
                row = target_df.iloc[0]
                labels[target] = int(row['label']) if row['label'] != -1 else -1
                kinetics[target] = {
                    'pKi': row.get('pKi', float('nan')),
                    'pIC50': row.get('pIC50', float('nan')),
                    'interaction_mode': row.get('interaction_mode', -1),
                    'kinetic_bias': row.get('kinetic_bias', float('nan')),
                }

        data = create_graph_data(smi, labels, kinetics, featurizer)
        if data is not None:
            graphs.append(data)

    return graphs


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Batch,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute multi-task loss with DAT upweighting."""
    losses = {}
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    mse_loss = nn.MSELoss()

    total_loss = 0
    n_tasks = 0

    # Target weights - DAT upweighted
    target_weights = {'DAT': DAT_WEIGHT, 'NET': 1.0, 'SERT': 1.0}

    for target in ['DAT', 'NET', 'SERT']:
        weight = target_weights[target]

        # Classification loss
        if target in outputs:
            logits = outputs[target]
            labels = getattr(batch, f'y_{target.lower()}', None)

            if labels is not None:
                labels = labels.to(device).long().view(-1)
                mask = labels != -1
                if mask.any():
                    loss_cls = ce_loss(logits[mask], labels[mask])
                    losses[f'{target}_cls'] = loss_cls
                    total_loss += loss_cls * weight
                    n_tasks += weight

        # pKi regression loss
        pki_key = f'{target}_pKi_mean'
        if pki_key in outputs:
            pred_pki = outputs[pki_key].view(-1)
            true_pki = getattr(batch, f'pKi_{target}', None)

            if true_pki is not None:
                true_pki = true_pki.to(device).view(-1)
                mask = ~torch.isnan(true_pki)
                if mask.any():
                    loss_pki = mse_loss(pred_pki[mask], true_pki[mask])
                    losses[f'{target}_pKi'] = loss_pki
                    total_loss += loss_pki * 0.5 * weight
                    n_tasks += 0.5 * weight

        # pIC50 regression loss
        pic50_key = f'{target}_pIC50_mean'
        if pic50_key in outputs:
            pred_pic50 = outputs[pic50_key].view(-1)
            true_pic50 = getattr(batch, f'pIC50_{target}', None)

            if true_pic50 is not None:
                true_pic50 = true_pic50.to(device).view(-1)
                mask = ~torch.isnan(true_pic50)
                if mask.any():
                    loss_pic50 = mse_loss(pred_pic50[mask], true_pic50[mask])
                    losses[f'{target}_pIC50'] = loss_pic50
                    total_loss += loss_pic50 * 0.5 * weight
                    n_tasks += 0.5 * weight

    losses['total'] = total_loss / max(n_tasks, 1)
    return losses


def train_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch, return_kinetics=True)
        losses = compute_loss(outputs, batch, device)

        loss = losses['total']
        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: PyGDataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    all_preds = {t: [] for t in ['DAT', 'NET', 'SERT']}
    all_labels = {t: [] for t in ['DAT', 'NET', 'SERT']}
    all_pki_preds = {t: [] for t in ['DAT', 'NET', 'SERT']}
    all_pki_true = {t: [] for t in ['DAT', 'NET', 'SERT']}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch, return_kinetics=True)

        for target in ['DAT', 'NET', 'SERT']:
            if target in outputs:
                probs = torch.softmax(outputs[target], dim=-1)
                substrate_prob = probs[:, 2] if probs.shape[1] > 2 else probs[:, 1]
                all_preds[target].extend(substrate_prob.cpu().numpy())

                labels = getattr(batch, f'y_{target.lower()}', None)
                if labels is not None:
                    all_labels[target].extend(labels.cpu().numpy())

            pki_key = f'{target}_pKi_mean'
            if pki_key in outputs:
                pred_pki = outputs[pki_key].view(-1).cpu().numpy()
                true_pki = getattr(batch, f'pKi_{target}', None)

                if true_pki is not None:
                    true_pki = true_pki.view(-1).cpu().numpy()
                    all_pki_preds[target].extend(pred_pki)
                    all_pki_true[target].extend(true_pki)

    metrics = {}

    for target in ['DAT', 'NET', 'SERT']:
        if all_preds[target] and all_labels[target]:
            preds = np.array(all_preds[target])
            labels = np.array(all_labels[target])
            mask = labels != -1
            if mask.sum() > 0:
                binary_labels = (labels[mask] == 2).astype(int)
                if len(np.unique(binary_labels)) > 1:
                    metrics[f'{target}_auc'] = roc_auc_score(binary_labels, preds[mask])

        if all_pki_preds[target] and all_pki_true[target]:
            preds = np.array(all_pki_preds[target])
            true = np.array(all_pki_true[target])
            mask = ~np.isnan(true)
            if mask.sum() > 10:
                metrics[f'{target}_pKi_r2'] = r2_score(true[mask], preds[mask])
                metrics[f'{target}_pKi_mae'] = mean_absolute_error(true[mask], preds[mask])

    aucs = [v for k, v in metrics.items() if k.endswith('_auc')]
    metrics['mean_auc'] = np.mean(aucs) if aucs else 0

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Kinetic Model Training v3 - Full 63k Dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/kinetic_splits', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='models/kinetic_v3', help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("KINETIC MODEL TRAINING v3 - 63k MOLECULES + DAT UPWEIGHT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"DAT weight: {DAT_WEIGHT}x")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge data
    print("\n" + "=" * 70)
    print("LOADING & MERGING DATA")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    train_df, val_df, test_df = merge_new_data(data_dir)

    train_df = load_kinetic_data(train_df, 'train')
    val_df = load_kinetic_data(val_df, 'val')
    test_df = load_kinetic_data(test_df, 'test')

    # Data stats
    print(f"\npKi availability (train):")
    for t in ['DAT', 'NET', 'SERT']:
        sub = train_df[train_df['target'] == t]
        pki_count = sub['pKi'].notna().sum()
        print(f"  {t}: {pki_count} / {len(sub)} ({100*pki_count/len(sub):.1f}%)")

    # Create graphs
    print("\nCreating molecular graphs...")
    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    train_graphs = prepare_dataset(train_df, featurizer, "Train")
    val_graphs = prepare_dataset(val_df, featurizer, "Val")
    test_graphs = prepare_dataset(test_df, featurizer, "Test")

    print(f"\nGraphs: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    # Model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    model = StereoGNNKinetic().to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_auc = 0
    start_epoch = 1
    patience = 20
    no_improve = 0

    # Resume from checkpoint if exists
    latest_ckpt = None
    for f in sorted(output_dir.glob('checkpoint_epoch_*.pt'), reverse=True):
        latest_ckpt = f
        break

    if latest_ckpt and latest_ckpt.exists():
        print(f"\nResuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('best_auc', ckpt.get('val_auc', 0))
        print(f"Resuming from epoch {ckpt['epoch']}, best AUC: {best_auc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        val_auc = val_metrics.get('mean_auc', 0)
        dat_auc = val_metrics.get('DAT_auc', 0)
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | DAT: {dat_auc:.4f} | LR: {lr:.6f}")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'best_auc': best_auc,
                'val_metrics': val_metrics,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"  -> Checkpoint saved")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_metrics': val_metrics,
            }, output_dir / 'best_kinetic_model.pt')
            print(f"  -> Saved best model (AUC: {val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(output_dir / 'best_kinetic_model.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\nTest Results:")
    for key, value in sorted(test_metrics.items()):
        print(f"  {key}: {value:.4f}")

    results = {
        'best_val_auc': best_auc,
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'dat_weight': DAT_WEIGHT,
        'total_train_molecules': len(train_graphs),
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best model: {output_dir / 'best_kinetic_model.pt'}")
    print(f"Results: {output_dir / 'results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
