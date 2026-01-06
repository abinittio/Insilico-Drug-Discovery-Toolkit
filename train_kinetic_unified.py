"""
Unified Kinetic Training Script
================================

Two-phase training:
1. Pretrain: Classification only (activity labels)
2. Finetune: Full kinetic model (classification + pKi + pIC50 + mode + bias)

Usage:
    python train_kinetic_unified.py --data-dir data/kinetic_splits
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from model import StereoGNNKinetic, count_parameters
from losses import KineticMultiTaskLoss
from dataset import (
    KineticTransporterDataset,
    create_kinetic_dataloaders,
    batch_to_kinetic_targets,
)
from config import CONFIG


def train_epoch(model, loader, optimizer, loss_fn, device, phase='full'):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Training ({phase})")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch, return_kinetics=True)

        # Convert batch to targets dict
        targets = batch_to_kinetic_targets(batch)

        # Compute loss
        if phase == 'pretrain':
            # Classification only
            loss = compute_classification_loss(outputs, batch, device)
        else:
            # Full kinetic loss
            losses = loss_fn(outputs, targets)
            loss = losses['total']

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(n_batches, 1)


def compute_classification_loss(outputs, targets, device):
    """Compute classification-only loss for pretraining."""
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0
    n_tasks = 0

    for target in ['DAT', 'NET', 'SERT']:
        if target in outputs:
            logits = outputs[target]
            labels = targets.get(target)

            if labels is not None:
                labels = labels.to(device).long()
                mask = labels != -1
                if mask.any():
                    loss = ce_loss(logits[mask], labels[mask])
                    total_loss += loss
                    n_tasks += 1

    return total_loss / max(n_tasks, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, phase='full'):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    all_preds = {t: [] for t in ['DAT', 'NET', 'SERT']}
    all_labels = {t: [] for t in ['DAT', 'NET', 'SERT']}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch, return_kinetics=True)
        targets = batch_to_kinetic_targets(batch)

        if phase == 'pretrain':
            loss = compute_classification_loss(outputs, batch, device)
        else:
            losses = loss_fn(outputs, targets)
            loss = losses['total']

        if not torch.isnan(loss):
            total_loss += loss.item()
            n_batches += 1

        # Collect predictions for AUC
        for target in ['DAT', 'NET', 'SERT']:
            if target in outputs:
                probs = torch.softmax(outputs[target], dim=-1)
                # Probability of being substrate (class 2)
                substrate_prob = probs[:, 2] if probs.shape[1] > 2 else probs[:, 1]
                all_preds[target].extend(substrate_prob.cpu().numpy())

                # Get labels from targets dict
                labels = targets.get(target)
                if labels is not None:
                    all_labels[target].extend(labels.cpu().numpy())

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for target in ['DAT', 'NET', 'SERT']:
        if all_preds[target] and all_labels[target]:
            preds = np.array(all_preds[target])
            labels = np.array(all_labels[target])
            # Filter out unknown labels
            mask = labels != -1
            if mask.sum() > 0:
                # Binary: substrate (2) vs not substrate (0, 1)
                binary_labels = (labels[mask] == 2).astype(int)
                if len(np.unique(binary_labels)) > 1:
                    aucs[target] = roc_auc_score(binary_labels, preds[mask])

    mean_auc = np.mean(list(aucs.values())) if aucs else 0

    return total_loss / max(n_batches, 1), mean_auc, aucs


def main():
    parser = argparse.ArgumentParser(description='Unified Kinetic Training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory with train/val/test.parquet')
    parser.add_argument('--epochs', type=int, default=130,
                        help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 60)
    print("Unified Kinetic Training")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    print("\nCreating StereoGNNKinetic model...")
    model = StereoGNNKinetic()
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Load data
    print(f"\nLoading data from {args.data_dir}")
    data_path = Path(args.data_dir)

    dataloaders = create_kinetic_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=0,  # Windows compatibility
        use_3d=False,
        augment=True,
    )

    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Kinetic loss function - trains on ALL available labels (masked multi-task)
    loss_fn = KineticMultiTaskLoss(
        learn_weights=True,
    ).to(device)

    # =========================================================================
    # TRAINING (All Available Labels - Masked Multi-Task)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING (Masked Multi-Task on All Available Labels)")
    print("=" * 60)
    print("  - Classification (activity labels)")
    print("  - pKi (where available)")
    print("  - pIC50 (where available)")
    print("  - Interaction mode (where available)")
    print("  - Kinetic bias (where available)")

    total_epochs = args.epochs
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_auc = 0
    patience = 20
    no_improve = 0

    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")

        train_loss = train_epoch(model, dataloaders['train'], optimizer, loss_fn, device, phase='full')
        val_loss, val_auc, val_aucs = evaluate(model, dataloaders['val'], loss_fn, device, phase='full')
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        if val_aucs:
            print(f"  Per-target: {' | '.join(f'{k}: {v:.4f}' for k, v in val_aucs.items())}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'val_auc': val_auc,
            }, output_dir / f'kinetic_model_{timestamp}.pt')
            # Also save as best
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
            }, output_dir / 'best_kinetic_model.pt')
            print(f"  -> Saved best model (AUC: {val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_kinetic_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_auc, test_aucs = evaluate(model, dataloaders['test'], loss_fn, device, phase='full')

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Mean AUC: {test_auc:.4f}")
    for target, auc in test_aucs.items():
        print(f"  {target} AUC: {auc:.4f}")

    # Save results
    results = {
        'best_val_auc': best_auc,
        'test_auc': test_auc,
        'test_aucs': test_aucs,
        'timestamp': timestamp,
    }

    with open(output_dir / f'kinetic_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best model saved to: {output_dir / 'best_kinetic_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
