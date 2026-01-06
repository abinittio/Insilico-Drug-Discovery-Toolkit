"""
Complete Training Pipeline
==========================

Two-stage training:
1. Pretrain on all transporter data
2. Fine-tune on monoamine-specific data

Handles:
- Data loading and preprocessing
- Model training with early stopping
- Checkpointing and logging
- Evaluation and metrics
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix
)
from tqdm import tqdm

from config import CONFIG
from dataset import TransporterDataset, get_dataloaders
from model_pretrain import (
    StereoGNNPretrain, StereoGNNFinetune,
    StereoGNNForAblation, count_parameters
)
from losses import FocalLoss, MultiTaskLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_pr_auc': [],
        }

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def save(self, filename: str = 'metrics.json'):
        with open(self.output_dir / filename, 'w') as f:
            json.dump(self.history, f, indent=2)


class PretrainTrainer:
    """
    Trainer for pretraining phase.

    Trains on all transporter data to learn general substrate representations.
    """

    def __init__(
        self,
        model: StereoGNNPretrain,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=CONFIG.training.learning_rate,
            weight_decay=CONFIG.training.weight_decay,
        )

        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=CONFIG.training.learning_rate,
            epochs=CONFIG.training.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=CONFIG.training.warmup_ratio,
        )

        # Loss
        self.loss_fn = FocalLoss(gamma=CONFIG.training.focal_gamma)

        # Tracking
        self.metrics = MetricsTracker(self.output_dir)
        self.early_stopping = EarlyStopping(
            patience=CONFIG.training.early_stopping_patience
        )

        self.best_auc = 0.0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass for all active targets
            outputs = self.model(batch)

            # Compute loss for each target
            loss = 0.0
            n_targets = 0

            for target in outputs:
                if target in ['node_attention', 'graph_embedding']:
                    continue

                # Get labels for this target
                if hasattr(batch, f'{target}_label'):
                    labels = getattr(batch, f'{target}_label')
                    mask = labels >= 0  # Valid labels only

                    if mask.sum() > 0:
                        target_loss = self.loss_fn(
                            outputs[target][mask],
                            labels[mask].long()
                        )
                        loss += target_loss
                        n_targets += 1

            if n_targets > 0:
                loss = loss / n_targets
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    CONFIG.training.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        all_preds = {t: [] for t in self.model.targets}
        all_labels = {t: [] for t in self.model.targets}
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc='Validating'):
            batch = batch.to(self.device)
            outputs = self.model(batch)

            for target in self.model.targets:
                if target not in outputs:
                    continue

                if hasattr(batch, f'{target}_label'):
                    labels = getattr(batch, f'{target}_label')
                    mask = labels >= 0

                    if mask.sum() > 0:
                        probs = F.softmax(outputs[target][mask], dim=-1)
                        all_preds[target].append(probs.cpu())
                        all_labels[target].append(labels[mask].cpu())

                        loss = self.loss_fn(
                            outputs[target][mask],
                            labels[mask].long()
                        )
                        total_loss += loss.item()

        # Compute metrics
        metrics = {'val_loss': total_loss / len(self.val_loader)}

        auc_scores = []
        for target in self.model.targets:
            if len(all_preds[target]) == 0:
                continue

            preds = torch.cat(all_preds[target], dim=0).numpy()
            labels = torch.cat(all_labels[target], dim=0).numpy()

            # ROC-AUC (one-vs-rest)
            try:
                auc_score = roc_auc_score(
                    labels, preds,
                    multi_class='ovr',
                    average='macro'
                )
                metrics[f'{target}_auc'] = auc_score
                auc_scores.append(auc_score)
            except:
                pass

        if auc_scores:
            metrics['mean_auc'] = np.mean(auc_scores)

        return metrics

    def train(self, num_epochs: int = None):
        """Full training loop."""
        num_epochs = num_epochs or CONFIG.training.num_epochs

        logger.info(f"Starting pretraining for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Parameters: {count_parameters(self.model):,}")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Val metrics: {val_metrics}")

            # Track
            self.metrics.update({
                'train_loss': train_loss,
                **val_metrics
            })
            self.metrics.save()

            # Save best model
            mean_auc = val_metrics.get('mean_auc', 0.0)
            if mean_auc > self.best_auc:
                self.best_auc = mean_auc
                self.save_checkpoint(f'best_pretrain.pt')
                logger.info(f"New best AUC: {mean_auc:.4f}")

            # Early stopping
            if self.early_stopping(mean_auc):
                logger.info("Early stopping triggered")
                break

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'pretrain_epoch_{epoch + 1}.pt')

        logger.info(f"\nPretraining complete. Best AUC: {self.best_auc:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'config': CONFIG.__dict__,
        }, self.output_dir / filename)


class FinetuneTrainer:
    """
    Trainer for fine-tuning phase.

    Fine-tunes on monoamine-specific data (DAT/NET/SERT).
    """

    def __init__(
        self,
        model: StereoGNNFinetune,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: Path,
        device: torch.device = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # Different learning rates for backbone vs heads
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.heads.parameters())

        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': CONFIG.training.learning_rate * 0.1},
            {'params': head_params, 'lr': CONFIG.training.learning_rate},
        ], weight_decay=CONFIG.training.weight_decay)

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )

        # Multi-task loss
        self.loss_fn = MultiTaskLoss(
            task_names=['DAT', 'NET', 'SERT'],
            loss_fn=FocalLoss(gamma=CONFIG.training.focal_gamma),
        )

        self.metrics = MetricsTracker(self.output_dir)
        self.early_stopping = EarlyStopping(
            patience=CONFIG.training.early_stopping_patience
        )

        self.best_auc = 0.0
        self.best_metrics = {}

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc='Fine-tuning')
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(batch)

            # Compute multi-task loss
            losses = {}
            for target in ['DAT', 'NET', 'SERT']:
                if hasattr(batch, f'{target}_label'):
                    labels = getattr(batch, f'{target}_label')
                    mask = labels >= 0

                    if mask.sum() > 0:
                        losses[target] = (outputs[target][mask], labels[mask].long())

            if losses:
                loss = self.loss_fn(losses)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    CONFIG.training.max_grad_norm
                )

                self.optimizer.step()
                total_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        self.scheduler.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Dict[str, float]:
        """Validate the model."""
        loader = loader or self.val_loader
        self.model.eval()

        all_preds = {t: [] for t in ['DAT', 'NET', 'SERT']}
        all_labels = {t: [] for t in ['DAT', 'NET', 'SERT']}
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch)

            for target in ['DAT', 'NET', 'SERT']:
                if hasattr(batch, f'{target}_label'):
                    labels = getattr(batch, f'{target}_label')
                    mask = labels >= 0

                    if mask.sum() > 0:
                        probs = F.softmax(outputs[target][mask], dim=-1)
                        all_preds[target].append(probs.cpu())
                        all_labels[target].append(labels[mask].cpu())

        # Compute metrics
        metrics = {}

        auc_scores = []
        pr_auc_scores = []

        for target in ['DAT', 'NET', 'SERT']:
            if len(all_preds[target]) == 0:
                continue

            preds = torch.cat(all_preds[target], dim=0).numpy()
            labels = torch.cat(all_labels[target], dim=0).numpy()

            # ROC-AUC
            try:
                auc_score = roc_auc_score(
                    labels, preds,
                    multi_class='ovr',
                    average='macro'
                )
                metrics[f'{target}_auc'] = auc_score
                auc_scores.append(auc_score)
            except:
                pass

            # PR-AUC for substrate class (class 2)
            try:
                binary_labels = (labels == 2).astype(int)
                substrate_probs = preds[:, 2]

                precision, recall, _ = precision_recall_curve(
                    binary_labels, substrate_probs
                )
                pr_auc = auc(recall, precision)
                metrics[f'{target}_pr_auc'] = pr_auc
                pr_auc_scores.append(pr_auc)
            except:
                pass

        if auc_scores:
            metrics['mean_auc'] = np.mean(auc_scores)
        if pr_auc_scores:
            metrics['mean_pr_auc'] = np.mean(pr_auc_scores)

        return metrics

    def evaluate_stereo_sensitivity(self) -> float:
        """
        Evaluate on stereoselective pairs.

        Returns percentage of pairs where the model correctly
        distinguishes between enantiomers.
        """
        from config import STEREOSELECTIVE_PAIRS
        from featurizer import MoleculeGraphFeaturizer
        from torch_geometric.data import Batch

        self.model.eval()
        featurizer = MoleculeGraphFeaturizer()

        correct = 0
        total = 0

        for pair in STEREOSELECTIVE_PAIRS:
            d_smiles = pair['d_isomer']
            l_smiles = pair['l_isomer']
            d_activity = pair['d_activity']
            l_activity = pair['l_activity']
            target = pair['target']

            # Featurize
            d_graph = featurizer.featurize(d_smiles)
            l_graph = featurizer.featurize(l_smiles)

            if d_graph is None or l_graph is None:
                continue

            d_batch = Batch.from_data_list([d_graph]).to(self.device)
            l_batch = Batch.from_data_list([l_graph]).to(self.device)

            with torch.no_grad():
                d_out = self.model(d_batch)
                l_out = self.model(l_batch)

                d_probs = F.softmax(d_out[target], dim=-1)[0]
                l_probs = F.softmax(l_out[target], dim=-1)[0]

                # Check if model distinguishes them
                d_pred = d_probs[2].item()  # Substrate probability
                l_pred = l_probs[2].item()

                # d-isomer is typically more active
                if d_activity > l_activity:
                    if d_pred > l_pred:
                        correct += 1
                else:
                    if l_pred > d_pred:
                        correct += 1

                total += 1

        return correct / total if total > 0 else 0.0

    def train(self, num_epochs: int = None):
        """Full fine-tuning loop."""
        num_epochs = num_epochs or CONFIG.training.finetune_epochs

        logger.info(f"Starting fine-tuning for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Val metrics: {val_metrics}")

            # Track
            self.metrics.update({
                'train_loss': train_loss,
                **val_metrics
            })
            self.metrics.save()

            # Check success criteria
            mean_auc = val_metrics.get('mean_auc', 0.0)

            if mean_auc > self.best_auc:
                self.best_auc = mean_auc
                self.best_metrics = val_metrics
                self.save_checkpoint('best_finetune.pt')
                logger.info(f"New best AUC: {mean_auc:.4f}")

            # Early stopping
            if self.early_stopping(mean_auc):
                logger.info("Early stopping triggered")
                break

        # Final evaluation on test set
        logger.info("\n" + "=" * 60)
        logger.info("Final Test Evaluation")
        logger.info("=" * 60)

        test_metrics = self.validate(self.test_loader)
        stereo_sensitivity = self.evaluate_stereo_sensitivity()

        logger.info(f"Test metrics: {test_metrics}")
        logger.info(f"Stereo sensitivity: {stereo_sensitivity:.2%}")

        # Save final results
        self.save_final_results(test_metrics, stereo_sensitivity)

        return test_metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_metrics': self.best_metrics,
        }, self.output_dir / filename)

    def save_final_results(
        self,
        test_metrics: Dict[str, float],
        stereo_sensitivity: float,
    ):
        """Save final evaluation results."""
        results = {
            'test_metrics': test_metrics,
            'stereo_sensitivity': stereo_sensitivity,
            'success_criteria': {
                'overall_auc_target': 0.85,
                'monoamine_auc_target': 0.95,
                'pr_auc_target': 0.65,
                'stereo_sensitivity_target': 0.80,
            },
            'pass': {
                'overall_auc': test_metrics.get('mean_auc', 0) >= 0.85,
                'pr_auc': test_metrics.get('mean_pr_auc', 0) >= 0.65,
                'stereo_sensitivity': stereo_sensitivity >= 0.80,
            },
            'timestamp': datetime.now().isoformat(),
        }

        with open(self.output_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)


def run_full_pipeline(
    pretrain_data_path: Path = None,
    finetune_data_path: Path = None,
    output_dir: Path = None,
    skip_pretrain: bool = False,
):
    """
    Run the complete training pipeline.

    Args:
        pretrain_data_path: Path to all transporter data
        finetune_data_path: Path to monoamine data
        output_dir: Output directory
        skip_pretrain: Skip pretraining (use existing checkpoint)
    """
    output_dir = output_dir or Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Stage 1: Pretraining
    if not skip_pretrain:
        logger.info("=" * 70)
        logger.info("STAGE 1: PRETRAINING ON ALL TRANSPORTERS")
        logger.info("=" * 70)

        # Load pretraining data
        # pretrain_loaders = get_dataloaders(pretrain_data_path, mode='pretrain')

        pretrain_model = StereoGNNPretrain()

        # pretrain_trainer = PretrainTrainer(
        #     model=pretrain_model,
        #     train_loader=pretrain_loaders['train'],
        #     val_loader=pretrain_loaders['val'],
        #     output_dir=output_dir / 'pretrain',
        #     device=device,
        # )
        # pretrain_trainer.train()

        logger.info("Pretraining complete (or skipped for now)")

    # Stage 2: Fine-tuning
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: FINE-TUNING ON MONOAMINE TRANSPORTERS")
    logger.info("=" * 70)

    # Create fine-tuning model
    pretrain_checkpoint = output_dir / 'pretrain' / 'best_pretrain.pt'

    if pretrain_checkpoint.exists():
        finetune_model = StereoGNNFinetune.from_pretrained(
            torch.load(pretrain_checkpoint)['model_state_dict'],
            freeze_backbone=False,
        )
    else:
        finetune_model = StereoGNNFinetune()

    # Load fine-tuning data
    # finetune_loaders = get_dataloaders(finetune_data_path, mode='finetune')

    # finetune_trainer = FinetuneTrainer(
    #     model=finetune_model,
    #     train_loader=finetune_loaders['train'],
    #     val_loader=finetune_loaders['val'],
    #     test_loader=finetune_loaders['test'],
    #     output_dir=output_dir / 'finetune',
    #     device=device,
    # )

    # final_metrics = finetune_trainer.train()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()
