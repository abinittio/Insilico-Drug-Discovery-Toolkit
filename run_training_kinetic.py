"""
Training Script for StereoGNN Kinetic Model
============================================

Trains the extended StereoGNN model for mechanistic kinetic parameter prediction.

Training Strategy:
1. Optional Stage 1: Pretrain on activity classification (transfer from base model)
2. Stage 2: Train full kinetic model with multi-task loss

Outputs:
- Activity classification (substrate/blocker/inactive)
- Binding affinity (pKi)
- Functional potency (pIC50)
- Interaction mode (substrate/competitive/non-competitive/partial)
- Kinetic bias (uptake preference)

Usage:
    python run_training_kinetic.py [--from-pretrained PATH] [--epochs N]
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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import StereoGNN, StereoGNNKinetic, count_parameters
from losses import KineticMultiTaskLoss
from dataset import (
    KineticTransporterDataset,
    create_kinetic_dataloaders,
    batch_to_kinetic_targets,
)
from config import CONFIG

# Try to import wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 25, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class KineticTrainer:
    """
    Trainer for StereoGNN Kinetic model.

    Handles:
    - Multi-task training (classification + regression)
    - Uncertainty-aware loss computation
    - Learning rate scheduling with warmup
    - Early stopping
    - Model checkpointing
    - Metrics logging
    """

    def __init__(
        self,
        model: StereoGNNKinetic,
        train_loader,
        val_loader,
        config=None,
        experiment_name: str = None,
        use_wandb: bool = False,
    ):
        self.config = config or CONFIG
        self.device = torch.device(self.config.training.device)
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Experiment tracking
        self.experiment_name = experiment_name or f"kinetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = self.config.data.models_dir / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = KineticMultiTaskLoss(
            learn_weights=self.config.kinetic.learn_task_weights,
            focal_gamma=self.config.training.focal_gamma,
        ).to(self.device)

        # Optimizer with separate learning rates
        self._setup_optimizer()

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.config.training.warmup_epochs,
            max_epochs=self.config.training.max_epochs,
            min_lr=self.config.training.min_lr,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            mode='max',
        )

        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = self.device.type == 'cuda'

        # Metrics storage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auroc': [],
            'val_pki_r2': [],
            'val_pic50_r2': [],
            'val_mode_acc': [],
            'learning_rate': [],
            'task_weights': [],
        }

        self.best_val_auroc = 0.0
        self.best_epoch = 0

        # Weights & Biases logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project="stereognn-kinetic",
                name=self.experiment_name,
                config={
                    'model': self.config.model.__dict__,
                    'training': self.config.training.__dict__,
                    'kinetic': self.config.kinetic.__dict__,
                }
            )

    def _setup_optimizer(self):
        """Setup optimizer with parameter groups."""
        # Different learning rates for different components
        param_groups = [
            {
                'params': self.model.node_encoder.parameters(),
                'lr': self.config.training.learning_rate,
                'name': 'node_encoder',
            },
            {
                'params': self.model.edge_encoder.parameters(),
                'lr': self.config.training.learning_rate,
                'name': 'edge_encoder',
            },
            {
                'params': self.model.gnn_layers.parameters(),
                'lr': self.config.training.learning_rate,
                'name': 'gnn_layers',
            },
            {
                'params': self.model.readout.parameters(),
                'lr': self.config.training.learning_rate,
                'name': 'readout',
            },
            {
                'params': self.model.shared_layer.parameters(),
                'lr': self.config.training.learning_rate * 2,
                'name': 'shared_layer',
            },
            {
                'params': self.model.heads.parameters(),
                'lr': self.config.training.learning_rate * 2,
                'name': 'classification_heads',
            },
            {
                'params': self.model.kinetic_heads.parameters(),
                'lr': self.config.training.learning_rate * 2,
                'name': 'kinetic_heads',
            },
            {
                'params': self.criterion.parameters(),
                'lr': self.config.training.learning_rate,
                'name': 'loss_weights',
            },
        ]

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        component_losses = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(batch, return_kinetics=True)
                    targets = batch_to_kinetic_targets(batch)
                    losses = self.criterion(output, targets)
                    loss = losses['total']

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch, return_kinetics=True)
                targets = batch_to_kinetic_targets(batch)
                losses = self.criterion(output, targets)
                loss = losses['total']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Accumulate component losses
            for key, value in losses.items():
                if key != 'total':
                    if key not in component_losses:
                        component_losses[key] = 0.0
                    component_losses[key] += value.item()

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in component_losses.items()}

        return {'loss': avg_loss, **avg_components}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # Collect predictions and targets
        all_preds = {task: {'class': [], 'pKi': [], 'pIC50': [], 'mode': []}
                     for task in ['DAT', 'NET', 'SERT']}
        all_targets = {task: {'class': [], 'pKi': [], 'pIC50': [], 'mode': []}
                       for task in ['DAT', 'NET', 'SERT']}

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = batch.to(self.device)

            output = self.model(batch, return_kinetics=True)
            targets = batch_to_kinetic_targets(batch)
            losses = self.criterion(output, targets)

            total_loss += losses['total'].item()
            num_batches += 1

            # Collect predictions
            for task in ['DAT', 'NET', 'SERT']:
                # Classification
                if task in output:
                    probs = torch.softmax(output[task], dim=-1)
                    all_preds[task]['class'].append(probs.cpu())
                    if f'{task}_class' in targets:
                        all_targets[task]['class'].append(targets[f'{task}_class'].cpu())

                # pKi
                if f'{task}_pKi_mean' in output and f'{task}_pKi' in targets:
                    all_preds[task]['pKi'].append(output[f'{task}_pKi_mean'].cpu())
                    all_targets[task]['pKi'].append(targets[f'{task}_pKi'].cpu())

                # pIC50
                if f'{task}_pIC50_mean' in output and f'{task}_pIC50' in targets:
                    all_preds[task]['pIC50'].append(output[f'{task}_pIC50_mean'].cpu())
                    all_targets[task]['pIC50'].append(targets[f'{task}_pIC50'].cpu())

                # Mode
                if f'{task}_interaction_mode' in output and f'{task}_mode' in targets:
                    mode_probs = torch.softmax(output[f'{task}_interaction_mode'], dim=-1)
                    all_preds[task]['mode'].append(mode_probs.cpu())
                    all_targets[task]['mode'].append(targets[f'{task}_mode'].cpu())

        # Compute metrics
        metrics = {'loss': total_loss / num_batches}

        # Classification AUROC
        aurocs = []
        for task in ['DAT', 'NET', 'SERT']:
            if all_preds[task]['class'] and all_targets[task]['class']:
                preds = torch.cat(all_preds[task]['class'], dim=0)
                targets_cat = torch.cat(all_targets[task]['class'], dim=0)

                # Filter valid labels
                mask = targets_cat >= 0
                if mask.sum() > 0:
                    auroc = self._compute_auroc(preds[mask], targets_cat[mask])
                    metrics[f'{task}_auroc'] = auroc
                    aurocs.append(auroc)

        if aurocs:
            metrics['mean_auroc'] = np.mean(aurocs)

        # Regression R²
        for task in ['DAT', 'NET', 'SERT']:
            for param in ['pKi', 'pIC50']:
                if all_preds[task][param] and all_targets[task][param]:
                    preds = torch.cat(all_preds[task][param], dim=0)
                    targets_cat = torch.cat(all_targets[task][param], dim=0)

                    # Filter valid (non-NaN)
                    mask = ~torch.isnan(targets_cat)
                    if mask.sum() > 10:
                        r2 = self._compute_r2(preds[mask], targets_cat[mask])
                        metrics[f'{task}_{param}_r2'] = r2

        # Mode accuracy
        mode_accs = []
        for task in ['DAT', 'NET', 'SERT']:
            if all_preds[task]['mode'] and all_targets[task]['mode']:
                preds = torch.cat(all_preds[task]['mode'], dim=0)
                targets_cat = torch.cat(all_targets[task]['mode'], dim=0)

                mask = targets_cat >= 0
                if mask.sum() > 0:
                    pred_classes = preds[mask].argmax(dim=-1)
                    acc = (pred_classes == targets_cat[mask]).float().mean().item()
                    metrics[f'{task}_mode_acc'] = acc
                    mode_accs.append(acc)

        if mode_accs:
            metrics['mean_mode_acc'] = np.mean(mode_accs)

        return metrics

    def _compute_auroc(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute AUROC for multi-class classification."""
        try:
            from sklearn.metrics import roc_auc_score
            preds_np = preds.numpy()
            targets_np = targets.numpy().astype(int)

            # One-vs-rest AUROC
            auroc = roc_auc_score(targets_np, preds_np, multi_class='ovr', average='macro')
            return auroc
        except Exception:
            return 0.5

    def _compute_r2(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute R² for regression."""
        ss_res = ((preds - targets) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return r2.item()

    def train(self, num_epochs: int = None) -> Dict[str, float]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.training.max_epochs
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()
            print(f"Train loss: {train_metrics['loss']:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"Val loss: {val_metrics['loss']:.4f}")
            if 'mean_auroc' in val_metrics:
                print(f"Val AUROC: {val_metrics['mean_auroc']:.4f}")

            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log task weights
            task_weights = self.criterion.get_task_weights()
            print(f"Task weights: {task_weights}")

            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auroc'].append(val_metrics.get('mean_auroc', 0))
            self.history['learning_rate'].append(current_lr)
            self.history['task_weights'].append(task_weights)

            # WandB logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_auroc': val_metrics.get('mean_auroc', 0),
                    'learning_rate': current_lr,
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    **{f'task_weight_{k}': v for k, v in task_weights.items()},
                })

            # Save best model
            val_auroc = val_metrics.get('mean_auroc', 0)
            if val_auroc > self.best_val_auroc:
                self.best_val_auroc = val_auroc
                self.best_epoch = epoch
                self._save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"New best model! AUROC: {val_auroc:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_metrics)

            # Early stopping
            if self.early_stopping(val_auroc):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Save final model and training history
        self._save_checkpoint('final_model.pt', epoch, val_metrics)
        self._save_history()

        print(f"\nTraining complete!")
        print(f"Best AUROC: {self.best_val_auroc:.4f} at epoch {self.best_epoch + 1}")

        return {
            'best_auroc': self.best_val_auroc,
            'best_epoch': self.best_epoch,
            'final_metrics': val_metrics,
        }

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'metrics': metrics,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
                'kinetic': self.config.kinetic.__dict__,
            },
        }
        torch.save(checkpoint, self.save_dir / filename)

    def _save_history(self):
        """Save training history."""
        history_df = pd.DataFrame({
            'epoch': range(len(self.history['train_loss'])),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_auroc': self.history['val_auroc'],
            'learning_rate': self.history['learning_rate'],
        })
        history_df.to_csv(self.save_dir / 'training_history.csv', index=False)

        # Save task weight history
        with open(self.save_dir / 'task_weights_history.json', 'w') as f:
            json.dump(self.history['task_weights'], f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train StereoGNN Kinetic Model')
    parser.add_argument('--from-pretrained', type=str, default=None,
                        help='Path to pretrained base StereoGNN model')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze GNN backbone during training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for saving')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory')

    args = parser.parse_args()

    print("=" * 60)
    print("StereoGNN Kinetic Model Training")
    print("=" * 60)

    # Create model
    if args.from_pretrained:
        print(f"\nLoading pretrained model from {args.from_pretrained}")
        checkpoint = torch.load(args.from_pretrained, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # Check if it's a kinetic model or base model by looking for kinetic-specific keys
        is_kinetic_model = any('kinetic' in k or 'pki_head' in k or 'pic50_head' in k for k in state_dict.keys())

        if is_kinetic_model:
            # Load directly as kinetic model
            model = StereoGNNKinetic()
            model.load_state_dict(state_dict)
            print("Loaded full kinetic model from checkpoint")
        else:
            # Load as base model and create kinetic model from it
            base_model = StereoGNN()
            base_model.load_state_dict(state_dict)
            model = StereoGNNKinetic.from_pretrained(
                base_model,
                freeze_backbone=args.freeze_backbone,
            )
            print("Created kinetic model from pretrained base")
    else:
        print("\nCreating new kinetic model (no pretraining)")
        model = StereoGNNKinetic()

    print(f"Model parameters: {count_parameters(model):,}")

    # Update config if needed
    if args.lr:
        CONFIG.training.learning_rate = args.lr
    if args.batch_size:
        CONFIG.training.batch_size = args.batch_size

    # Create dataloaders
    data_path = Path(args.data_dir) if args.data_dir else CONFIG.data.data_dir
    print(f"\nLoading data from {data_path}")

    try:
        dataloaders = create_kinetic_dataloaders(
            data_path=data_path,
            batch_size=args.batch_size or CONFIG.training.batch_size,
            num_workers=CONFIG.training.num_workers,
            use_3d=False,
            augment=True,
        )
        print(f"Train samples: {len(dataloaders['train'].dataset)}")
        print(f"Val samples: {len(dataloaders['val'].dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nNote: You need to prepare kinetic training data with columns:")
        print("  - smiles: SMILES string")
        print("  - target: DAT/NET/SERT")
        print("  - label: Activity class (0=inactive, 1=blocker, 2=substrate)")
        print("  - pKi: Binding affinity (optional)")
        print("  - pIC50: Functional potency (optional)")
        print("  - interaction_mode: 0-3 (optional)")
        print("  - kinetic_bias: 0-1 (optional)")
        return

    # Create trainer
    trainer = KineticTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=CONFIG,
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
    )

    # Train
    num_epochs = args.epochs or CONFIG.training.max_epochs
    results = trainer.train(num_epochs=num_epochs)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best AUROC: {results['best_auroc']:.4f}")
    print(f"Best epoch: {results['best_epoch'] + 1}")
    print(f"Model saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()
