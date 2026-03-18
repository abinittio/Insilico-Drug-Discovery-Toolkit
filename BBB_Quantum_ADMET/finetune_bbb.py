"""
Quantum BBB Fine-tuning Script
Binary classification using 34-dimensional quantum features

Run AFTER pretraining completes.
Target: Beat 0.8968 AUC (stereo-only baseline) -> Push to 0.9+

Usage: python finetune_bbb.py
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Optional, Tuple, List

from model import QuantumAwareEncoder, count_parameters
from quantum_features import QuantumFeatureExtractor

logger = logging.getLogger(__name__)


class BBBQuantumClassifier(nn.Module):
    """BBB classifier with quantum-aware encoder."""

    def __init__(self, encoder: QuantumAwareEncoder, hidden_dim: int = 256, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head (encoder outputs hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and classification head."""
        if self.freeze_encoder:
            with torch.no_grad():
                graph_embed = self.encoder(x, edge_index, batch)
        else:
            graph_embed = self.encoder(x, edge_index, batch)
        return self.classifier(graph_embed)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for fine-tuning."""
        self.freeze_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True


def load_bbb_data() -> Tuple[Optional[List], Optional[np.ndarray]]:
    """Load BBB dataset and convert to quantum graphs."""
    # Try multiple paths
    bbb_paths = [
        'data/bbbp_dataset.csv',
        '../BBB_System/data/bbbp_dataset.csv',
    ]

    bbb_path = None
    for p in bbb_paths:
        if os.path.exists(p):
            bbb_path = p
            break

    if bbb_path is None:
        logger.error("BBBP dataset not found!")
        logger.error("Please copy bbbp_dataset.csv to data/ folder")
        return None, None

    logger.info("Loading BBB data from %s...", bbb_path)
    df = pd.read_csv(bbb_path)
    logger.info("  Total molecules: %d", len(df))
    logger.info("  BBB+ (permeable): %d", df['BBB_permeability'].sum())
    logger.info("  BBB- (non-permeable): %d", len(df) - df['BBB_permeability'].sum())

    # Convert to quantum graphs
    logger.info("Converting to quantum graphs (34 features)...")
    logger.info("Using ETKDG for 3D conformers...")
    sys.stdout.flush()

    extractor = QuantumFeatureExtractor(use_etkdg=True)

    graphs = []
    labels = []

    for idx, row in df.iterrows():
        smiles = row['SMILES']
        label = float(row['BBB_permeability'])

        graph = extractor.mol_to_graph(smiles)

        if graph is not None and graph.x.shape[1] == 34:
            graph.y = torch.tensor([label], dtype=torch.float)
            graphs.append(graph)
            labels.append(label)

        if (idx + 1) % 500 == 0:
            logger.info("  Processed %d/%d (%d valid)", idx + 1, len(df), len(graphs))
            sys.stdout.flush()

    logger.info("Valid graphs: %d/%d", len(graphs), len(df))
    return graphs, np.array(labels)


def train_epoch(
    model: BBBQuantumClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.view(-1), batch.y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(out).detach().cpu().numpy().flatten())
        all_labels.extend(batch.y.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc


def evaluate(
    model: BBBQuantumClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float, List[float], List[float]]:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.view(-1), batch.y.view(-1))

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(out).cpu().numpy().flatten())
            all_labels.extend(batch.y.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)

    return total_loss / len(loader), auc, acc, all_preds, all_labels


def main() -> None:
    """Run BBB fine-tuning with cross-validation."""
    logger.info("QUANTUM BBB FINE-TUNING")
    logger.info("34-dimensional features | Target: Beat 0.8968 AUC")
    logger.info("Started: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRETRAINED_PATH = 'models/pretrained_quantum_encoder.pth'
    BATCH_SIZE = 32
    EPOCHS_FROZEN = 15      # Train classifier with frozen encoder
    EPOCHS_FINETUNE = 25    # Fine-tune everything
    LR_FROZEN = 0.001
    LR_FINETUNE = 0.0001
    N_FOLDS = 5

    logger.info("Device: %s", DEVICE)
    logger.info("Training: %d frozen + %d fine-tune epochs", EPOCHS_FROZEN, EPOCHS_FINETUNE)

    # Check for pretrained encoder
    if not os.path.exists(PRETRAINED_PATH):
        logger.warning("Pretrained encoder not found at %s", PRETRAINED_PATH)
        logger.warning("Training from scratch (results may be worse)")
        use_pretrained = False
    else:
        use_pretrained = True
        logger.info("Using pretrained encoder: %s", PRETRAINED_PATH)

    # Load BBB data
    graphs, labels = load_bbb_data()
    if graphs is None:
        return

    # 5-fold cross-validation
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_fold_aucs = []
    all_fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs, labels)):
        logger.info("FOLD %d/%d", fold + 1, N_FOLDS)

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

        logger.info("Train: %d, Val: %d", len(train_graphs), len(val_graphs))

        # Create encoder
        encoder = QuantumAwareEncoder(
            node_features=34,
            hidden_dim=256,
            num_layers=5,
            num_heads=8,
            dropout=0.2
        )

        # Load pretrained weights
        if use_pretrained:
            encoder.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
            logger.info("Loaded pretrained encoder weights")

        # Create classifier
        model = BBBQuantumClassifier(encoder, hidden_dim=256, freeze_encoder=True).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        best_val_auc = 0
        best_epoch = 0

        # Phase 1: Train with frozen encoder
        logger.info("Phase 1: Training classifier (encoder frozen)...")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_FROZEN,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FROZEN)

        for epoch in range(1, EPOCHS_FROZEN + 1):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                marker = " *BEST*"
                torch.save(model.state_dict(), f'models/bbb_quantum_fold{fold+1}_best.pth')

            if epoch % 5 == 0 or marker:
                logger.info("  Epoch %2d | Train AUC: %.4f | Val AUC: %.4f | Acc: %.4f%s",
                            epoch, train_auc, val_auc, val_acc, marker)
            sys.stdout.flush()

        # Phase 2: Fine-tune entire model
        logger.info("Phase 2: Fine-tuning entire model...")
        model.unfreeze_encoder()

        optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)

        for epoch in range(1, EPOCHS_FINETUNE + 1):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = EPOCHS_FROZEN + epoch
                marker = " *BEST*"
                torch.save(model.state_dict(), f'models/bbb_quantum_fold{fold+1}_best.pth')

            if epoch % 5 == 0 or marker:
                logger.info("  Epoch %2d | Train AUC: %.4f | Val AUC: %.4f | Acc: %.4f%s",
                            epoch, train_auc, val_auc, val_acc, marker)
            sys.stdout.flush()

        # Load best and evaluate
        model.load_state_dict(torch.load(f'models/bbb_quantum_fold{fold+1}_best.pth', map_location=DEVICE))
        _, final_auc, final_acc, preds, true_labels = evaluate(model, val_loader, criterion, DEVICE)

        all_fold_aucs.append(final_auc)
        all_fold_accs.append(final_acc)

        preds_binary = (np.array(preds) > 0.5).astype(int)
        precision = precision_score(true_labels, preds_binary)
        recall = recall_score(true_labels, preds_binary)
        f1 = f1_score(true_labels, preds_binary)

        logger.info("Fold %d Results (Best @ Epoch %d):", fold + 1, best_epoch)
        logger.info("  AUC:       %.4f", final_auc)
        logger.info("  Accuracy:  %.4f", final_acc)
        logger.info("  Precision: %.4f", precision)
        logger.info("  Recall:    %.4f", recall)
        logger.info("  F1:        %.4f", f1)

    # Final summary
    logger.info("FINAL RESULTS (5-FOLD CROSS-VALIDATION)")
    logger.info("Mean AUC:      %.4f +/- %.4f", np.mean(all_fold_aucs), np.std(all_fold_aucs))
    logger.info("Mean Accuracy: %.4f +/- %.4f", np.mean(all_fold_accs), np.std(all_fold_accs))
    logger.info("Per-fold AUCs: %s", [f'{auc:.4f}' for auc in all_fold_aucs])

    # Compare to baselines
    STEREO_BASELINE = 0.8968
    mean_auc = np.mean(all_fold_aucs)

    logger.info("COMPARISON TO BASELINES")
    logger.info("Stereo-only baseline (21 features): 0.8968")
    logger.info("Quantum model (34 features):        %.4f", mean_auc)

    if mean_auc > STEREO_BASELINE:
        improvement = (mean_auc - STEREO_BASELINE) * 100
        logger.info("SUCCESS! Beat stereo baseline by %.2f%%", improvement)
    else:
        diff = (STEREO_BASELINE - mean_auc) * 100
        logger.info("Did not beat stereo baseline (diff: -%.2f%%)", diff)

    if mean_auc >= 0.9:
        logger.info("ACHIEVED 0.9+ AUC TARGET!")

    logger.info("Completed: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Models saved: models/bbb_quantum_fold*_best.pth")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    main()
