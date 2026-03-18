#!/usr/bin/env python
"""
StereoGNN Training Script - ORDINAL REGRESSION VERSION
========================================================

Outputs a continuous "Transport Activity Score" from 0 to 1:
  - 0.0-0.33: Inactive (no significant interaction)
  - 0.33-0.66: Blocker (inhibits without being transported)
  - 0.66-1.0: Substrate (actively transported)

The score is interpretable as a probability of substrate-like behavior.

Usage:
    python run_training_ordinal.py --small --finetune-only
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

from config import CONFIG, STEREOSELECTIVE_PAIRS
from featurizer import MoleculeGraphFeaturizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_ordinal.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# ORDINAL REGRESSION LOSS
# =============================================================================

class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression using cumulative thresholds.

    For K ordered classes (0 < 1 < 2), we learn K-1 thresholds.
    P(Y > k) = sigmoid(score - threshold_k)

    This ensures monotonicity and gives interpretable probabilities.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        # Learnable thresholds (K-1 for K classes)
        # Initialize so that thresholds are roughly evenly spaced
        self.thresholds = nn.Parameter(torch.tensor([-0.5, 0.5]))

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        scores: (N,) - raw scores from model
        labels: (N,) - ordinal labels 0, 1, 2
        """
        # Ensure thresholds are ordered
        sorted_thresholds = torch.sort(self.thresholds)[0]

        # For each sample, compute cumulative probabilities
        # P(Y > k) for k = 0, 1
        loss = 0.0

        for k in range(self.num_classes - 1):
            # P(Y > k) = sigmoid(score - threshold_k)
            prob_greater_k = torch.sigmoid(scores - sorted_thresholds[k])

            # Target: 1 if label > k, else 0
            target = (labels > k).float()

            # Binary cross entropy
            bce = -target * torch.log(prob_greater_k + 1e-8) - (1 - target) * torch.log(1 - prob_greater_k + 1e-8)
            loss = loss + bce.mean()

        return loss / (self.num_classes - 1)

    def predict_proba(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert raw scores to class probabilities."""
        sorted_thresholds = torch.sort(self.thresholds)[0]

        # P(Y > k) for each threshold
        cumulative_probs = torch.sigmoid(scores.unsqueeze(-1) - sorted_thresholds)

        # P(Y = k) = P(Y > k-1) - P(Y > k)
        # P(Y = 0) = 1 - P(Y > 0)
        # P(Y = 1) = P(Y > 0) - P(Y > 1)
        # P(Y = 2) = P(Y > 1)

        p0 = 1 - cumulative_probs[:, 0]
        p1 = cumulative_probs[:, 0] - cumulative_probs[:, 1]
        p2 = cumulative_probs[:, 1]

        return torch.stack([p0, p1, p2], dim=-1)

    def predict_score(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert raw scores to 0-1 transport activity score."""
        # Use class probabilities as weights
        probs = self.predict_proba(scores)
        # Expected value: 0*P(0) + 0.5*P(1) + 1.0*P(2)
        activity_score = 0.0 * probs[:, 0] + 0.5 * probs[:, 1] + 1.0 * probs[:, 2]
        return activity_score


class SmoothOrdinalLoss(nn.Module):
    """
    Simpler ordinal loss: MSE to ordinal target values.
    Maps labels 0->0.0, 1->0.5, 2->1.0 and uses smooth L1 loss.
    """

    def __init__(self):
        super().__init__()
        self.label_to_score = {0: 0.0, 1: 0.5, 2: 1.0}

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        scores: (N,) - predicted activity scores (0-1)
        labels: (N,) - ordinal labels 0, 1, 2
        """
        # Convert labels to continuous targets
        targets = torch.zeros_like(scores)
        for label_val, score_val in self.label_to_score.items():
            targets[labels == label_val] = score_val

        # Smooth L1 loss (Huber loss) - robust to outliers
        loss = F.smooth_l1_loss(scores, targets)

        # Add ranking loss component for better ordering
        # Penalize cases where higher labels have lower scores
        ranking_loss = self._ranking_loss(scores, labels)

        return loss + 0.3 * ranking_loss

    def _ranking_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Pairwise ranking loss to enforce ordering."""
        n = scores.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=scores.device)

        # Sample pairs
        n_pairs = min(n * 2, 100)  # Limit number of pairs
        idx1 = torch.randint(0, n, (n_pairs,), device=scores.device)
        idx2 = torch.randint(0, n, (n_pairs,), device=scores.device)

        # For pairs where label1 > label2, we want score1 > score2
        label_diff = labels[idx1] - labels[idx2]
        score_diff = scores[idx1] - scores[idx2]

        # Margin ranking loss: want score_diff to have same sign as label_diff
        # max(0, -label_sign * score_diff + margin)
        margin = 0.1
        label_sign = torch.sign(label_diff.float())
        loss = F.relu(-label_sign * score_diff + margin)

        # Only count pairs where labels differ
        mask = label_diff != 0
        if mask.sum() > 0:
            return loss[mask].mean()
        return torch.tensor(0.0, device=scores.device)


# =============================================================================
# ORDINAL REGRESSION MODEL
# =============================================================================

class StereoGNNOrdinal(nn.Module):
    """
    StereoGNN with ordinal regression output.

    Outputs a single continuous score per target (0-1).
    Higher score = more substrate-like behavior.
    """

    MONOAMINE_TARGETS = ['DAT', 'NET', 'SERT']

    def __init__(
        self,
        node_dim: int = 86,
        edge_dim: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = GATv2Conv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout,
                edge_dim=64, concat=True,
            )
            self.gnn_layers.append(conv)

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Shared readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Task-specific heads outputting SINGLE SCORE (not 3 classes)
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(hidden_dim, 96),
                nn.LayerNorm(96),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(48, 1),  # Single output
                nn.Sigmoid(),  # Constrain to 0-1
            )
            for target in self.MONOAMINE_TARGETS
        })

    def forward(self, data, return_embedding: bool = False) -> Dict[str, torch.Tensor]:
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for i, conv in enumerate(self.gnn_layers):
            x_new = conv(x, data.edge_index, edge_attr)
            x_new = self.norms[i](x_new)
            x = F.relu(x_new) + x

        graph_emb = global_mean_pool(x, data.batch)
        graph_emb = self.readout(graph_emb)

        output = {}
        for target in self.MONOAMINE_TARGETS:
            # Output is (N, 1), squeeze to (N,)
            output[target] = self.heads[target](graph_emb).squeeze(-1)

        if return_embedding:
            output['graph_embedding'] = graph_emb

        return output

    @classmethod
    def from_classification_model(cls, classification_model, node_dim: int = 86, edge_dim: int = 18):
        """Initialize from a trained classification model, keeping backbone weights."""
        ordinal_model = cls(node_dim=node_dim, edge_dim=edge_dim)

        # Copy backbone weights
        ordinal_model.node_encoder.load_state_dict(classification_model.node_encoder.state_dict())
        ordinal_model.edge_encoder.load_state_dict(classification_model.edge_encoder.state_dict())
        ordinal_model.readout.load_state_dict(classification_model.readout.state_dict())

        for i in range(len(ordinal_model.gnn_layers)):
            ordinal_model.gnn_layers[i].load_state_dict(classification_model.gnn_layers[i].state_dict())
            ordinal_model.norms[i].load_state_dict(classification_model.norms[i].state_dict())

        logger.info("Initialized ordinal model from classification backbone")
        return ordinal_model


# =============================================================================
# DATA LOADING (reuse from classification)
# =============================================================================

def load_finetuning_data() -> pd.DataFrame:
    """Load monoamine-specific data for training."""
    logger.info("Loading training data...")

    cache_file = Path("./data/final_augmented.parquet")
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        logger.info(f"Loaded {len(df)} records from cache")
        return df

    # Fallback to generating data
    from data_comprehensive import ComprehensiveLiteratureData
    from data_sar_expansion import SARExpander, DecoyGenerator
    from data_augmentation import StereoisomerEnumerator

    lit_df = ComprehensiveLiteratureData.get_all_data()
    expander = SARExpander()
    sar_df = expander.generate_all()
    decoy_gen = DecoyGenerator()
    decoy_df = decoy_gen.generate_decoys()

    combined = pd.concat([lit_df, sar_df, decoy_df], ignore_index=True)

    enumerator = StereoisomerEnumerator()
    augmented = enumerator.augment_dataset(combined)

    return augmented


def create_graph_dataset(
    df: pd.DataFrame,
    featurizer: MoleculeGraphFeaturizer,
    targets: List[str],
) -> List[Data]:
    """Convert DataFrame to list of PyG Data objects."""
    graphs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        smiles = row['smiles']
        target = row['target']
        label = row['label']

        if target not in targets:
            continue

        graph = featurizer.featurize(smiles)
        if graph is None:
            continue

        # Store label
        setattr(graph, f'{target}_label', torch.tensor([label], dtype=torch.long))

        # -1 for other targets
        for other_target in targets:
            if other_target != target:
                if not hasattr(graph, f'{other_target}_label'):
                    setattr(graph, f'{other_target}_label', torch.tensor([-1], dtype=torch.long))

        graphs.append(graph)

    return graphs


def collate_fn(data_list: List[Data]) -> Batch:
    """Custom collate function."""
    batch = Batch.from_data_list(data_list)

    for target in ['DAT', 'NET', 'SERT']:
        labels = []
        for data in data_list:
            if hasattr(data, f'{target}_label'):
                labels.append(getattr(data, f'{target}_label'))
            else:
                labels.append(torch.tensor([-1], dtype=torch.long))
        setattr(batch, f'{target}_label', torch.cat(labels))

    return batch


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    targets: List[str],
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch)

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_tasks = 0

        for target in targets:
            if target not in outputs:
                continue
            labels = getattr(batch, f'{target}_label')
            mask = labels >= 0

            if mask.sum() > 0:
                # Convert labels to continuous targets
                continuous_labels = labels[mask].float()
                continuous_labels = continuous_labels / 2.0  # 0->0, 1->0.5, 2->1.0

                target_loss = F.smooth_l1_loss(outputs[target][mask], continuous_labels)
                loss = loss + target_loss
                n_tasks += 1

        if n_tasks > 0:
            loss = loss / n_tasks
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader) if len(loader) > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    targets: List[str],
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    all_preds = {t: [] for t in targets}
    all_labels = {t: [] for t in targets}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        for target in targets:
            if target not in outputs:
                continue
            labels = getattr(batch, f'{target}_label')
            mask = labels >= 0

            if mask.sum() > 0:
                all_preds[target].append(outputs[target][mask].cpu().numpy())
                all_labels[target].append(labels[mask].cpu().numpy())

    metrics = {}
    mse_scores = []
    corr_scores = []

    for target in targets:
        if len(all_preds[target]) == 0:
            continue

        preds = np.concatenate(all_preds[target])
        labels = np.concatenate(all_labels[target])

        # Convert labels to continuous (0, 0.5, 1.0)
        continuous_labels = labels / 2.0

        # MSE
        mse = mean_squared_error(continuous_labels, preds)
        metrics[f'{target}_mse'] = mse
        mse_scores.append(mse)

        # MAE
        mae = mean_absolute_error(continuous_labels, preds)
        metrics[f'{target}_mae'] = mae

        # Spearman correlation (rank-based)
        if len(np.unique(labels)) > 1:
            spearman, _ = spearmanr(labels, preds)
            metrics[f'{target}_spearman'] = spearman
            corr_scores.append(spearman)

        # Kendall tau (ordinal consistency)
        if len(np.unique(labels)) > 1:
            kendall, _ = kendalltau(labels, preds)
            metrics[f'{target}_kendall'] = kendall

        # Classification accuracy (using 0.33, 0.66 thresholds)
        pred_classes = np.digitize(preds, [0.33, 0.66])
        accuracy = (pred_classes == labels).mean()
        metrics[f'{target}_accuracy'] = accuracy

        # ROC-AUC for substrate detection
        try:
            binary_labels = (labels == 2).astype(int)
            if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                auc_score = roc_auc_score(binary_labels, preds)
                metrics[f'{target}_auc'] = auc_score
        except Exception:
            pass

    if mse_scores:
        metrics['mean_mse'] = np.mean(mse_scores)
    if corr_scores:
        metrics['mean_spearman'] = np.mean(corr_scores)

    return metrics


def evaluate_stereo_sensitivity(
    model: nn.Module,
    featurizer: MoleculeGraphFeaturizer,
    device: torch.device,
) -> Tuple[float, List[Dict]]:
    """Test on known stereoselective pairs."""
    model.eval()

    correct = 0
    total = 0
    results = []

    for pair in STEREOSELECTIVE_PAIRS:
        d_smiles = pair['d_isomer']
        l_smiles = pair['l_isomer']
        target = pair['target']
        d_activity = pair['d_activity']
        l_activity = pair['l_activity']

        d_graph = featurizer.featurize(d_smiles)
        l_graph = featurizer.featurize(l_smiles)

        if d_graph is None or l_graph is None:
            continue

        d_batch = Batch.from_data_list([d_graph]).to(device)
        l_batch = Batch.from_data_list([l_graph]).to(device)

        with torch.no_grad():
            d_out = model(d_batch)
            l_out = model(l_batch)

            d_score = d_out[target].item()
            l_score = l_out[target].item()

            # Check if ordering is correct
            if d_activity > l_activity:
                is_correct = d_score > l_score
            else:
                is_correct = l_score > d_score

            if is_correct:
                correct += 1

            total += 1

            result = {
                'name': pair.get('name', f'{target} pair'),
                'd_score': d_score,
                'l_score': l_score,
                'correct': is_correct,
                'margin': d_score - l_score,
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            logger.info(f"  {result['name']}: d={d_score:.3f} l={l_score:.3f} {status}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main(args):
    """Main training pipeline for ordinal regression."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Device: {device}")

    # Create featurizer
    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    # Get feature dimensions
    from rdkit import Chem
    sample_graph = featurizer.featurize("CC(N)Cc1ccccc1")
    node_dim = sample_graph.x.shape[1]
    edge_dim = sample_graph.edge_attr.shape[1]
    logger.info(f"Feature dimensions: node={node_dim}, edge={edge_dim}")

    # Load data
    df = load_finetuning_data()

    # Create graphs
    logger.info("Creating molecular graphs...")
    targets = ['DAT', 'NET', 'SERT']
    graphs = create_graph_dataset(df, featurizer, targets)
    logger.info(f"Created {len(graphs)} graphs")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    n_train = int(0.8 * len(graphs))
    n_val = int(0.1 * len(graphs))

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train+n_val]]
    test_graphs = [graphs[i] for i in indices[n_train+n_val:]]

    logger.info(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # Create loaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Create model
    if args.from_classification:
        # Load classification model and initialize ordinal from it
        from run_training import StereoGNNSmallFinetune
        class_model = StereoGNNSmallFinetune(node_dim=node_dim, edge_dim=edge_dim)
        checkpoint = torch.load(args.from_classification, map_location=device, weights_only=False)
        class_model.load_state_dict(checkpoint['model_state_dict'])
        model = StereoGNNOrdinal.from_classification_model(class_model, node_dim, edge_dim)
    else:
        model = StereoGNNOrdinal(node_dim=node_dim, edge_dim=edge_dim)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training
    logger.info("\n" + "="*70)
    logger.info("TRAINING ORDINAL REGRESSION MODEL")
    logger.info("="*70)

    best_spearman = -1
    best_metrics = {}
    patience = 20
    no_improve = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, None, device, targets)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, targets)

        mean_spearman = val_metrics.get('mean_spearman', 0)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"loss={train_loss:.4f}, spearman={mean_spearman:.4f}, "
                   f"DAT={val_metrics.get('DAT_spearman', 0):.3f}, "
                   f"NET={val_metrics.get('NET_spearman', 0):.3f}, "
                   f"SERT={val_metrics.get('SERT_spearman', 0):.3f}")

        if mean_spearman > best_spearman:
            best_spearman = mean_spearman
            best_metrics = val_metrics.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_spearman': best_spearman,
                'metrics': best_metrics,
            }, output_dir / 'best_model_ordinal.pt')
            logger.info(f"  -> New best model saved!")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            break

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model_ordinal.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)

    test_metrics = evaluate(model, test_loader, device, targets)
    logger.info("Test Metrics:")
    for k, v in sorted(test_metrics.items()):
        logger.info(f"  {k}: {v:.4f}")

    # Stereo sensitivity
    logger.info("\nStereo Sensitivity Test:")
    stereo_sens, stereo_results = evaluate_stereo_sensitivity(model, featurizer, device)
    logger.info(f"Stereo sensitivity: {stereo_sens:.2%}")

    test_metrics['stereo_sensitivity'] = stereo_sens

    # Save results
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'stereo_results': stereo_results,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'results_ordinal.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("="*70)
    logger.info("ORDINAL TRAINING COMPLETE")
    logger.info("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StereoGNN with Ordinal Regression")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training")
    parser.add_argument("--from-classification", type=str, default=None,
                       help="Initialize from classification model checkpoint")

    args = parser.parse_args()
    main(args)
