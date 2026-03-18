"""
Training Pipeline for StereoGNN
================================

Complete training loop with:
- Cosine warmup learning rate schedule
- Early stopping
- Gradient clipping
- Mixed precision training
- Checkpoint saving/loading
- Wandb/Tensorboard logging
- Multi-task loss handling
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm

from model import StereoGNN, StereoGNNForAblation
from losses import MultiTaskLoss, compute_class_weights
from dataset import create_dataloaders, TransporterDataset
from config import CONFIG, TrainingConfig


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.base_lrs[i] * lr_scale, self.min_lr)

    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping to terminate training when validation loss plateaus."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Complete training pipeline for StereoGNN.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig = None,
        experiment_name: str = "stereo_gnn",
        use_wandb: bool = False,
    ):
        self.config = config or CONFIG.training
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb

        # Create output directories
        self.output_dir = CONFIG.data.models_dir / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_auroc = 0.0
        self.training_history = defaultdict(list)

        # Initialize components (will be set in setup)
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = None
        self.early_stopping = None

    def setup(
        self,
        train_dataset: TransporterDataset,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Set up optimizer, scheduler, and loss function."""
        # Compute class weights if not provided
        if class_weights is None and self.config.auto_class_weights:
            class_weights = train_dataset.get_class_weights()

        # Move class weights to device
        if class_weights:
            class_weights = {
                k: v.to(self.device) for k, v in class_weights.items()
            }

        # Loss function
        self.loss_fn = MultiTaskLoss(
            loss_fn=self.config.loss_type,
            learn_weights=True,
            focal_gamma=self.config.focal_gamma,
            class_weights=class_weights,
            ignore_index=-1,
        ).to(self.device)

        # Optimizer
        # Separate learning rates for different components
        param_groups = [
            {
                'params': self.model.node_encoder.parameters(),
                'lr': self.config.learning_rate,
            },
            {
                'params': self.model.gnn_layers.parameters(),
                'lr': self.config.learning_rate,
            },
            {
                'params': self.model.heads.parameters(),
                'lr': self.config.learning_rate * 2,  # Higher LR for heads
            },
        ]

        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=self.config.max_epochs,
            min_lr=self.config.min_lr,
        )

        # Mixed precision
        self.scaler = GradScaler()

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode='max',  # Maximize AUROC
        )

        print(f"Training setup complete:")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  Scheduler: Cosine with warmup")
        print(f"  Loss: {self.config.loss_type}")
        if class_weights:
            for k, v in class_weights.items():
                print(f"  {k} weights: {v.cpu().numpy()}")

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = defaultdict(list)
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass with mixed precision
            with autocast():
                predictions = self.model(batch)

                targets = {
                    'DAT': batch.y_dat,
                    'NET': batch.y_net,
                    'SERT': batch.y_sert,
                }

                losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Record losses
            for k, v in losses.items():
                epoch_losses[k].append(v.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'DAT': f"{losses['DAT'].item():.4f}",
            })

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        all_predictions = {task: [] for task in ['DAT', 'NET', 'SERT']}
        all_targets = {task: [] for task in ['DAT', 'NET', 'SERT']}
        epoch_losses = defaultdict(list)

        for batch in dataloader:
            batch = batch.to(self.device)

            with autocast():
                predictions = self.model(batch)

                targets = {
                    'DAT': batch.y_dat,
                    'NET': batch.y_net,
                    'SERT': batch.y_sert,
                }

                losses = self.loss_fn(predictions, targets)

            for k, v in losses.items():
                epoch_losses[k].append(v.item())

            # Collect predictions and targets for metrics
            for task in all_predictions:
                pred_probs = torch.softmax(predictions[task], dim=-1)
                all_predictions[task].append(pred_probs.cpu())
                all_targets[task].append(targets[task].cpu())

        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}

        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        avg_losses.update(metrics)

        return avg_losses

    def _compute_metrics(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        metrics = {}

        for task in predictions:
            preds = torch.cat(predictions[task], dim=0).numpy()
            targs = torch.cat(targets[task], dim=0).numpy()

            # Filter valid labels
            valid_mask = targs >= 0
            if valid_mask.sum() < 10:
                continue

            preds = preds[valid_mask]
            targs = targs[valid_mask]

            # Binary: substrate vs non-substrate
            targs_binary = (targs == 2).astype(int)
            preds_substrate = preds[:, 2]  # Probability of substrate

            try:
                auroc = roc_auc_score(targs_binary, preds_substrate)
                prauc = average_precision_score(targs_binary, preds_substrate)
                metrics[f'{task}_auroc'] = auroc
                metrics[f'{task}_prauc'] = prauc
            except Exception as e:
                print(f"Warning: Could not compute metrics for {task}: {e}")

        # Compute average AUROC across all tasks
        aurocs = [v for k, v in metrics.items() if 'auroc' in k]
        if aurocs:
            metrics['avg_auroc'] = np.mean(aurocs)

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.max_epochs

        print("=" * 60)
        print(f"Starting training for {num_epochs} epochs")
        print("=" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Record history
            for k, v in train_metrics.items():
                self.training_history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.training_history[f'val_{k}'].append(v)
            self.training_history['lr'].append(self.scheduler.get_lr()[0])

            # Print progress
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f} | "
                f"Val AUROC: {val_metrics.get('avg_auroc', 0):.4f} | "
                f"LR: {self.scheduler.get_lr()[0]:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Check for best model
            current_auroc = val_metrics.get('avg_auroc', 0)
            if current_auroc > self.best_val_auroc:
                self.best_val_auroc = current_auroc
                self.save_checkpoint('best_model.pt')
                print(f"  -> New best model! AUROC: {current_auroc:.4f}")

            # Early stopping
            if self.early_stopping(current_auroc):
                print(f"Early stopping at epoch {epoch}")
                break

            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Save final model
        self.save_checkpoint('final_model.pt')

        # Save training history
        self.save_history()

        return dict(self.training_history)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auroc': self.best_val_auroc,
            'training_history': dict(self.training_history),
            'config': {
                'model': CONFIG.model.__dict__,
                'training': CONFIG.training.__dict__,
            },
        }
        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auroc = checkpoint['best_val_auroc']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(dict(self.training_history), f, indent=2)


def train_model(
    experiment_name: str = "stereo_gnn_v1",
    use_ablation: bool = False,
    **kwargs,
) -> Trainer:
    """
    Convenience function to train a model from scratch.

    Args:
        experiment_name: Name for the experiment
        use_ablation: If True, train ablation model (no stereo)
        **kwargs: Override config parameters

    Returns:
        Trainer object with trained model
    """
    # Create model
    if use_ablation:
        model = StereoGNNForAblation()
        experiment_name = f"{experiment_name}_ablation"
    else:
        model = StereoGNN()

    # Create dataloaders
    dataloaders = create_dataloaders(
        batch_size=CONFIG.training.batch_size,
        num_workers=CONFIG.training.num_workers,
        use_3d=False,  # Disable 3D for faster training initially
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        experiment_name=experiment_name,
    )

    # Setup training
    trainer.setup(dataloaders['train'].dataset)

    # Train
    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
    )

    return trainer


if __name__ == "__main__":
    print("=" * 60)
    print("StereoGNN Training Pipeline Test")
    print("=" * 60)

    # Create small test to verify pipeline works
    from data_curation import DataCurationPipeline

    # Run data curation
    pipeline = DataCurationPipeline()
    splits = pipeline.run(use_cache=False)

    # Create model
    model = StereoGNN()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(model, experiment_name="test_run")

    # Create dataloaders
    dataloaders = create_dataloaders(
        batch_size=8,
        num_workers=0,
        use_3d=False,
    )

    # Setup
    trainer.setup(dataloaders['train'].dataset)

    # Run a few epochs
    if len(dataloaders['train'].dataset) > 0:
        history = trainer.train(
            dataloaders['train'],
            dataloaders['val'],
            num_epochs=3,
        )

        print("\nTraining complete!")
        print(f"Best validation AUROC: {trainer.best_val_auroc:.4f}")
