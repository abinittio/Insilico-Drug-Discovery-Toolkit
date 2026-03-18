"""
Training Script for Quantum ADMET Multi-Task Model

Supports:
- Self-supervised pretraining on ZINC
- Multi-task supervised fine-tuning on ADMET endpoints
- Checkpoint saving and resumption
- Early stopping

Run: python train.py --mode pretrain
     python train.py --mode finetune
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle

from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG
from model import QuantumADMETModel, QuantumPretrainingModel, count_parameters
from quantum_features import QuantumFeatureExtractor, batch_smiles_to_graphs


def pretrain_epoch(
    model: QuantumPretrainingModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Run one pretraining epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        pred_mw, pred_ac, pred_logp, pred_stereo = model(
            batch.x, batch.edge_index, batch.batch
        )

        # Targets
        target_mw = batch.mol_weight.view(-1, 1)
        target_ac = batch.num_atoms.view(-1, 1)
        target_logp = batch.logp.view(-1, 1)

        # Check for stereo target
        if hasattr(batch, 'has_stereo'):
            target_stereo = batch.has_stereo.view(-1, 1)
        else:
            # Infer from features (feature 29 = is_chiral_center)
            # Average over atoms in each graph
            target_stereo = torch.zeros(pred_stereo.shape, device=device)

        # Losses
        loss_mw = mse(pred_mw, target_mw)
        loss_ac = mse(pred_ac, target_ac)
        loss_logp = mse(pred_logp, target_logp)
        loss_stereo = bce(pred_stereo, target_stereo)

        loss = loss_mw + loss_ac + loss_logp + 0.5 * loss_stereo

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG.gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def finetune_epoch(
    model: QuantumADMETModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    task_weights: Dict[str, float],
    device: str
) -> Tuple[float, Dict[str, float]]:
    """Run one fine-tuning epoch."""
    model.train()
    total_loss = 0
    task_losses = {task: 0.0 for task in task_weights.keys()}
    num_batches = 0

    mse = nn.MSELoss(reduction='none')

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        predictions = model(batch.x, batch.edge_index, batch.batch)

        # Multi-task loss
        loss = 0
        for task, weight in task_weights.items():
            if hasattr(batch, task):
                target = getattr(batch, task).view(-1, 1)
                pred = predictions[task]

                # Mask NaN targets
                mask = ~torch.isnan(target)
                if mask.any():
                    task_loss = mse(pred[mask], target[mask]).mean()
                    loss = loss + weight * task_loss
                    task_losses[task] += task_loss.item()

        if isinstance(loss, int) and loss == 0:
            continue  # No valid targets in this batch

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG.gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, task_losses

    avg_loss = total_loss / num_batches
    task_losses = {k: v / num_batches for k, v in task_losses.items()}

    return avg_loss, task_losses


def evaluate(
    model: QuantumADMETModel,
    loader: DataLoader,
    task_weights: Dict[str, float],
    device: str
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = {task: [] for task in task_weights.keys()}
    all_targets = {task: [] for task in task_weights.keys()}

    mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            predictions = model(batch.x, batch.edge_index, batch.batch)

            loss = 0
            for task, weight in task_weights.items():
                if hasattr(batch, task):
                    target = getattr(batch, task).view(-1, 1)
                    pred = predictions[task]

                    mask = ~torch.isnan(target)
                    if mask.any():
                        task_loss = mse(pred[mask], target[mask]).mean()
                        loss = loss + weight * task_loss

                        all_preds[task].extend(pred[mask].cpu().numpy().flatten())
                        all_targets[task].extend(target[mask].cpu().numpy().flatten())

            if not isinstance(loss, int):
                total_loss += loss.item()
                num_batches += 1

    if num_batches == 0:
        return 0.0, {}

    avg_loss = total_loss / num_batches

    # Compute per-task metrics
    metrics = {}
    for task in task_weights.keys():
        if len(all_preds[task]) > 0:
            preds = np.array(all_preds[task])
            targets = np.array(all_targets[task])

            metrics[task] = {
                'mse': mean_squared_error(targets, preds),
                'rmse': np.sqrt(mean_squared_error(targets, preds)),
                'mae': mean_absolute_error(targets, preds),
                'r2': r2_score(targets, preds) if len(targets) > 1 else 0.0
            }

    return avg_loss, metrics


def pretrain(args):
    """Self-supervised pretraining on ZINC."""
    print("=" * 70)
    print("QUANTUM ADMET - SELF-SUPERVISED PRETRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {TRAINING_CONFIG.device}")
    print()

    # Check for cached graphs
    cache_path = os.path.join(DATA_CONFIG.data_dir, 'zinc_quantum_graphs.pkl')

    if os.path.exists(cache_path) and not args.regenerate:
        print(f"Loading cached graphs from {cache_path}...")
        with open(cache_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
    else:
        print("Generating quantum graphs from ZINC...")
        # Load ZINC SMILES
        zinc_path = os.path.join(DATA_CONFIG.data_dir, 'zinc250k.csv')

        if os.path.exists(zinc_path):
            df = pd.read_csv(zinc_path)
            smiles_list = df['smiles'].tolist()[:args.num_molecules]
        else:
            print(f"ZINC file not found at {zinc_path}")
            print("Please provide ZINC data or use --zinc-path")
            return

        print(f"Processing {len(smiles_list)} molecules...")
        graphs = batch_smiles_to_graphs(
            smiles_list,
            use_etkdg=args.use_etkdg,
            verbose=True
        )

        # Add stereo labels
        for graph in graphs:
            # Check if any atom has chiral features
            has_stereo = (graph.x[:, 28] > 0).any().float()  # Feature 29 (0-indexed 28)
            graph.has_stereo = torch.tensor([has_stereo], dtype=torch.float)

        # Save cache
        os.makedirs(DATA_CONFIG.data_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"Saved {len(graphs)} graphs to {cache_path}")

    # Create data loader
    loader = DataLoader(
        graphs,
        batch_size=TRAINING_CONFIG.pretrain_batch_size,
        shuffle=True
    )

    # Create model
    model = QuantumPretrainingModel(
        node_features=MODEL_CONFIG.node_features,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        num_gat_layers=MODEL_CONFIG.num_gat_layers,
        num_heads=MODEL_CONFIG.num_heads,
        dropout=MODEL_CONFIG.dropout
    ).to(TRAINING_CONFIG.device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG.pretrain_lr,
        weight_decay=TRAINING_CONFIG.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG.pretrain_epochs
    )

    # Training loop
    os.makedirs(DATA_CONFIG.checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    print(f"\nTraining for {TRAINING_CONFIG.pretrain_epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, TRAINING_CONFIG.pretrain_epochs + 1):
        loss = pretrain_epoch(model, loader, optimizer, TRAINING_CONFIG.device)
        scheduler.step()

        marker = ""
        if loss < best_loss:
            best_loss = loss
            marker = " *BEST*"
            torch.save(
                model.state_dict(),
                os.path.join(DATA_CONFIG.model_dir, 'pretrained_quantum.pth')
            )
            torch.save(
                model.encoder.state_dict(),
                os.path.join(DATA_CONFIG.model_dir, 'pretrained_quantum_encoder.pth')
            )

        print(f"Epoch {epoch:3d}/{TRAINING_CONFIG.pretrain_epochs} | "
              f"Loss: {loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}{marker}")
        sys.stdout.flush()

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(DATA_CONFIG.checkpoint_dir, f'pretrain_epoch_{epoch:03d}.pth')
            )

    print("-" * 60)
    print(f"Pretraining complete! Best loss: {best_loss:.6f}")
    print(f"Encoder saved to: {DATA_CONFIG.model_dir}/pretrained_quantum_encoder.pth")


def finetune(args):
    """Fine-tune on ADMET endpoints."""
    print("=" * 70)
    print("QUANTUM ADMET - MULTI-TASK FINE-TUNING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {TRAINING_CONFIG.device}")
    print()

    # Load ADMET data
    admet_path = os.path.join(DATA_CONFIG.data_dir, 'admet_combined.csv')

    if not os.path.exists(admet_path):
        print(f"ADMET data not found at {admet_path}")
        print("Please prepare ADMET dataset first.")
        return

    df = pd.read_csv(admet_path)
    print(f"Loaded {len(df)} molecules with ADMET data")

    # Convert to graphs with targets
    print("Converting to quantum graphs...")
    smiles_list = df['smiles'].tolist()

    targets_list = []
    for _, row in df.iterrows():
        targets = {}
        for endpoint in DATA_CONFIG.endpoints:
            if endpoint in row and pd.notna(row[endpoint]):
                targets[endpoint] = float(row[endpoint])
        targets_list.append(targets)

    graphs = batch_smiles_to_graphs(
        smiles_list,
        targets_list=targets_list,
        use_etkdg=args.use_etkdg,
        verbose=True
    )

    # K-fold cross-validation
    kfold = KFold(n_splits=TRAINING_CONFIG.n_folds, shuffle=True, random_state=42)

    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs)):
        print("\n" + "=" * 60)
        print(f"FOLD {fold + 1}/{TRAINING_CONFIG.n_folds}")
        print("=" * 60)

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_loader = DataLoader(
            train_graphs,
            batch_size=TRAINING_CONFIG.finetune_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_graphs,
            batch_size=TRAINING_CONFIG.finetune_batch_size
        )

        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

        # Create model
        model = QuantumADMETModel(
            node_features=MODEL_CONFIG.node_features,
            hidden_dim=MODEL_CONFIG.hidden_dim,
            num_gat_layers=MODEL_CONFIG.num_gat_layers,
            num_heads=MODEL_CONFIG.num_heads,
            num_tasks=MODEL_CONFIG.num_tasks,
            dropout=MODEL_CONFIG.dropout
        ).to(TRAINING_CONFIG.device)

        # Load pretrained encoder if available
        pretrained_path = os.path.join(DATA_CONFIG.model_dir, 'pretrained_quantum_encoder.pth')
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained encoder from {pretrained_path}")
            model.encoder.load_state_dict(
                torch.load(pretrained_path, map_location=TRAINING_CONFIG.device)
            )

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG.finetune_lr,
            weight_decay=TRAINING_CONFIG.weight_decay
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=TRAINING_CONFIG.finetune_epochs
        )

        # Training
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, TRAINING_CONFIG.finetune_epochs + 1):
            train_loss, train_task_losses = finetune_epoch(
                model, train_loader, optimizer,
                TRAINING_CONFIG.task_weights, TRAINING_CONFIG.device
            )

            val_loss, val_metrics = evaluate(
                model, val_loader,
                TRAINING_CONFIG.task_weights, TRAINING_CONFIG.device
            )

            scheduler.step()

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                marker = " *BEST*"
                torch.save(
                    model.state_dict(),
                    os.path.join(DATA_CONFIG.model_dir, f'admet_fold{fold+1}_best.pth')
                )
            else:
                patience_counter += 1

            if epoch % 10 == 0 or marker:
                # Format metrics
                metric_str = " | ".join([
                    f"{t[:4]}: {m['rmse']:.3f}"
                    for t, m in val_metrics.items()
                ])
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | {metric_str}{marker}")
                sys.stdout.flush()

            # Early stopping
            if patience_counter >= TRAINING_CONFIG.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model and get final metrics
        model.load_state_dict(
            torch.load(os.path.join(DATA_CONFIG.model_dir, f'admet_fold{fold+1}_best.pth'),
                      map_location=TRAINING_CONFIG.device)
        )
        _, final_metrics = evaluate(
            model, val_loader,
            TRAINING_CONFIG.task_weights, TRAINING_CONFIG.device
        )

        print(f"\nFold {fold + 1} Final Results:")
        for task, metrics in final_metrics.items():
            print(f"  {task}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

        all_fold_metrics.append(final_metrics)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS (CROSS-VALIDATION)")
    print("=" * 70)

    for task in DATA_CONFIG.endpoints:
        rmses = [m[task]['rmse'] for m in all_fold_metrics if task in m]
        r2s = [m[task]['r2'] for m in all_fold_metrics if task in m]

        if rmses:
            print(f"{task:10s}: RMSE = {np.mean(rmses):.4f} +/- {np.std(rmses):.4f} | "
                  f"R2 = {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description='Train Quantum ADMET Model')
    parser.add_argument('--mode', choices=['pretrain', 'finetune'], required=True,
                        help='Training mode')
    parser.add_argument('--num-molecules', type=int, default=50000,
                        help='Number of ZINC molecules for pretraining')
    parser.add_argument('--use-etkdg', action='store_true', default=True,
                        help='Use ETKDG for 3D conformers')
    parser.add_argument('--no-etkdg', action='store_false', dest='use_etkdg',
                        help='Disable ETKDG')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerate cached graphs')

    args = parser.parse_args()

    # Create directories
    os.makedirs(DATA_CONFIG.data_dir, exist_ok=True)
    os.makedirs(DATA_CONFIG.model_dir, exist_ok=True)
    os.makedirs(DATA_CONFIG.checkpoint_dir, exist_ok=True)
    os.makedirs(DATA_CONFIG.results_dir, exist_ok=True)

    if args.mode == 'pretrain':
        pretrain(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
