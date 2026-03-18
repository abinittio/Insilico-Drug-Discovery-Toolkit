"""
Quantum BBB Fine-tuning Script
Binary classification using 34-dimensional quantum features

Run AFTER pretraining completes.
Target: Beat 0.8968 AUC (stereo-only baseline) → Push to 0.9+

Usage: python finetune_bbb.py
"""

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

from model import QuantumAwareEncoder, count_parameters
from quantum_features import QuantumFeatureExtractor


class BBBQuantumClassifier(nn.Module):
    """BBB classifier with quantum-aware encoder."""

    def __init__(self, encoder, hidden_dim=256, freeze_encoder=False):
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

    def forward(self, x, edge_index, batch):
        if self.freeze_encoder:
            with torch.no_grad():
                graph_embed = self.encoder(x, edge_index, batch)
        else:
            graph_embed = self.encoder(x, edge_index, batch)
        return self.classifier(graph_embed)

    def unfreeze_encoder(self):
        self.freeze_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True


def load_bbb_data():
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
        print("ERROR: BBBP dataset not found!")
        print("Please copy bbbp_dataset.csv to data/ folder")
        return None, None

    print(f"Loading BBB data from {bbb_path}...")
    df = pd.read_csv(bbb_path)
    print(f"  Total molecules: {len(df)}")
    print(f"  BBB+ (permeable): {df['BBB_permeability'].sum()}")
    print(f"  BBB- (non-permeable): {len(df) - df['BBB_permeability'].sum()}")

    # Convert to quantum graphs
    print("\nConverting to quantum graphs (34 features)...")
    print("Using ETKDG for 3D conformers...")
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
            print(f"  Processed {idx+1}/{len(df)} ({len(graphs)} valid)")
            sys.stdout.flush()

    print(f"Valid graphs: {len(graphs)}/{len(df)}")
    return graphs, np.array(labels)


def train_epoch(model, loader, optimizer, criterion, device):
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


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
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


def main():
    print("=" * 70)
    print("QUANTUM BBB FINE-TUNING")
    print("34-dimensional features | Target: Beat 0.8968 AUC")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRETRAINED_PATH = 'models/pretrained_quantum_encoder.pth'
    BATCH_SIZE = 32
    EPOCHS_FROZEN = 15      # Train classifier with frozen encoder
    EPOCHS_FINETUNE = 25    # Fine-tune everything
    LR_FROZEN = 0.001
    LR_FINETUNE = 0.0001
    N_FOLDS = 5

    print(f"Device: {DEVICE}")
    print(f"Training: {EPOCHS_FROZEN} frozen + {EPOCHS_FINETUNE} fine-tune epochs")
    print()

    # Check for pretrained encoder
    if not os.path.exists(PRETRAINED_PATH):
        print(f"WARNING: Pretrained encoder not found at {PRETRAINED_PATH}")
        print("Training from scratch (results may be worse)")
        use_pretrained = False
    else:
        use_pretrained = True
        print(f"Using pretrained encoder: {PRETRAINED_PATH}")

    # Load BBB data
    graphs, labels = load_bbb_data()
    if graphs is None:
        return

    print()

    # 5-fold cross-validation
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_fold_aucs = []
    all_fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(graphs, labels)):
        print("=" * 60)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print("=" * 60)

        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)

        print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

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
            print("Loaded pretrained encoder weights")

        # Create classifier
        model = BBBQuantumClassifier(encoder, hidden_dim=256, freeze_encoder=True).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        best_val_auc = 0
        best_epoch = 0

        # Phase 1: Train with frozen encoder
        print(f"\nPhase 1: Training classifier (encoder frozen)...")
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
                print(f"  Epoch {epoch:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f}{marker}")
            sys.stdout.flush()

        # Phase 2: Fine-tune entire model
        print(f"\nPhase 2: Fine-tuning entire model...")
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
                print(f"  Epoch {epoch:2d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f}{marker}")
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

        print(f"\nFold {fold+1} Results (Best @ Epoch {best_epoch}):")
        print(f"  AUC:       {final_auc:.4f}")
        print(f"  Accuracy:  {final_acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print()

    # Final summary
    print("=" * 70)
    print("FINAL RESULTS (5-FOLD CROSS-VALIDATION)")
    print("=" * 70)
    print(f"Mean AUC:      {np.mean(all_fold_aucs):.4f} +/- {np.std(all_fold_aucs):.4f}")
    print(f"Mean Accuracy: {np.mean(all_fold_accs):.4f} +/- {np.std(all_fold_accs):.4f}")
    print()
    print(f"Per-fold AUCs: {[f'{auc:.4f}' for auc in all_fold_aucs]}")
    print()

    # Compare to baselines
    STEREO_BASELINE = 0.8968
    mean_auc = np.mean(all_fold_aucs)

    print("-" * 40)
    print("COMPARISON TO BASELINES")
    print("-" * 40)
    print(f"Stereo-only baseline (21 features): 0.8968")
    print(f"Quantum model (34 features):        {mean_auc:.4f}")

    if mean_auc > STEREO_BASELINE:
        improvement = (mean_auc - STEREO_BASELINE) * 100
        print(f"\nSUCCESS! Beat stereo baseline by {improvement:.2f}%")
    else:
        diff = (STEREO_BASELINE - mean_auc) * 100
        print(f"\nDid not beat stereo baseline (diff: -{diff:.2f}%)")

    if mean_auc >= 0.9:
        print("ACHIEVED 0.9+ AUC TARGET!")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Models saved: models/bbb_quantum_fold*_best.pth")


if __name__ == "__main__":
    main()
