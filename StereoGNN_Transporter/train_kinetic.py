"""
StereoGNN Kinetic Model Training Script
========================================

Simple training script optimized for running on Anaconda.

Usage:
    # Basic training (uses defaults)
    python train_kinetic.py

    # Custom epochs
    python train_kinetic.py --epochs 50

    # Full options
    python train_kinetic.py --epochs 100 --batch-size 16 --lr 0.0001

    # Quick test run
    python train_kinetic.py --test-run
"""

import logging
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from model import StereoGNNKinetic
from dataset import KineticTransporterDataset, create_kinetic_dataloaders, batch_to_kinetic_targets
from losses import KineticMultiTaskLoss
from featurizer import MoleculeGraphFeaturizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train StereoGNN Kinetic Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Early stopping
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # Paths
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data'),
                        help='Directory containing training data')
    parser.add_argument('--save-dir', type=str, default=str(PROJECT_ROOT / 'models'),
                        help='Directory to save trained models')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to train on')

    # Testing
    parser.add_argument('--test-run', action='store_true',
                        help='Quick test run with 3 epochs')

    return parser.parse_args()


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def get_device(device_arg: str) -> torch.device:
    """Get the device to train on."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU (no GPU detected)")
    else:
        device = torch.device(device_arg)
        logger.info(f"Using device: {device}")
    return device


def load_data(data_dir: str, batch_size: int):
    """Load training, validation and test data."""
    logger.info("\n" + "=" * 50)
    logger.info("Loading Data")
    logger.info("=" * 50)

    data_path = Path(data_dir)

    # Check for training data
    train_file = data_path / 'train.parquet'
    if not train_file.exists():
        logger.error(f"Training data not found at {train_file}")
        logger.error("Please run data curation first: python data_curation_kinetic.py")
        sys.exit(1)

    logger.info(f"Loading from {data_path}")

    # Create featurizer
    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    # Create datasets - KineticTransporterDataset loads from data_path/{split}.parquet
    train_dataset = KineticTransporterDataset(
        data_path=data_path,
        split='train',
        featurizer=featurizer,
        pre_featurize=True,
        use_3d=False,
    )

    val_dataset = KineticTransporterDataset(
        data_path=data_path,
        split='val',
        featurizer=featurizer,
        pre_featurize=True,
        use_3d=False,
    ) if (data_path / 'val.parquet').exists() else None

    test_dataset = KineticTransporterDataset(
        data_path=data_path,
        split='test',
        featurizer=featurizer,
        pre_featurize=True,
        use_3d=False,
    ) if (data_path / 'test.parquet').exists() else None

    logger.info(f"  Train samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"  Val samples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"  Test samples: {len(test_dataset)}")

    # Create dataloaders
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Get targets
        targets = batch_to_kinetic_targets(batch)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, loader, criterion, device):
    """Validate the model."""
    if loader is None:
        return float('inf'), {}

    model.eval()
    total_loss = 0
    num_batches = 0
    all_losses = {}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            outputs = model(batch)
            targets = batch_to_kinetic_targets(batch)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total']

            total_loss += loss.item()
            num_batches += 1

            # Accumulate individual losses
            for key, val in loss_dict.items():
                if key not in all_losses:
                    all_losses[key] = 0
                if isinstance(val, torch.Tensor):
                    all_losses[key] += val.item()
                else:
                    all_losses[key] += val

    # Average losses
    avg_loss = total_loss / max(num_batches, 1)
    for key in all_losses:
        all_losses[key] /= max(num_batches, 1)

    return avg_loss, all_losses


def main():
    """Main training loop."""
    args = parse_args()

    # Handle test run
    if args.test_run:
        args.epochs = 3
        logger.info("\n*** TEST RUN MODE (3 epochs) ***\n")

    logger.info("=" * 50)
    logger.info("StereoGNN Kinetic Model Training")
    logger.info("=" * 50)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Patience: {args.patience}")

    # Setup device
    device = get_device(args.device)

    # Load data
    train_loader, val_loader, test_loader = load_data(args.data_dir, args.batch_size)

    # Create model
    logger.info("\n" + "=" * 50)
    logger.info("Creating Model")
    logger.info("=" * 50)

    model = StereoGNNKinetic(config=CONFIG.model)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Create loss function
    criterion = KineticMultiTaskLoss(learn_weights=True)
    criterion = criterion.to(device)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("\n" + "=" * 50)
    logger.info("Training")
    logger.info("=" * 50)

    best_val_loss = float('inf')
    best_epoch = 0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_losses = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Time
        epoch_time = time.time() - epoch_start

        # Print progress
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"LR: {lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, save_dir / 'best_kinetic_model.pt')
            logger.info(f"  --> Saved best model (val_loss: {val_loss:.4f})")

        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"\nEarly stopping at epoch {epoch}")
            break

    # Training complete
    total_time = time.time() - training_start
    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {save_dir / 'best_kinetic_model.pt'}")

    # Final evaluation on test set
    if test_loader is not None and len(test_loader.dataset) > 1:
        logger.info("\n" + "=" * 50)
        logger.info("Test Set Evaluation")
        logger.info("=" * 50)

        # Load best model
        try:
            checkpoint = torch.load(save_dir / 'best_kinetic_model.pt', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            test_loss, test_losses = validate(model, test_loader, criterion, device)
            logger.info(f"Test loss: {test_loss:.4f}")

            for key, val in test_losses.items():
                logger.info(f"  {key}: {val:.4f}")
        except Exception as e:
            logger.info(f"Test evaluation skipped due to: {e}")
    else:
        logger.info("\nTest set too small for evaluation (need >1 sample)")

    logger.info("\n" + "=" * 50)
    logger.info("Done!")
    logger.info("=" * 50)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    main()
