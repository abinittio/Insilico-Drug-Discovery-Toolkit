"""
Cardiac Ion Channel Panel Model
================================
Full CiPA panel: hERG + Nav1.5 + Cav1.2

Multi-task model for complete cardiac safety prediction.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from model import StereoGNN, count_parameters
from featurizer import MoleculeGraphFeaturizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CARDIAC_TARGETS = ['hERG', 'Nav1.5', 'Cav1.2']
FP_DIM = 1024 + 8


def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_arr = np.zeros(1024)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolWt(mol) / 500,
        Descriptors.NumHDonors(mol) / 5,
        Descriptors.NumHAcceptors(mol) / 10,
        Descriptors.NumRotatableBonds(mol) / 10,
        Descriptors.NumAromaticRings(mol) / 5,
        Descriptors.FractionCSP3(mol),
    ]

    return np.concatenate([fp_arr, desc]).astype(np.float32)


class CardiacPanelClassifier(nn.Module):
    """Multi-task cardiac ion channel classifier with FP fusion."""

    def __init__(self, backbone, hidden_dim=128, fp_dim=FP_DIM, n_targets=3, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.n_targets = n_targets

        # FP encoder
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = hidden_dim + hidden_dim

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-target heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, 2),
            ) for _ in range(n_targets)
        ])

    def forward(self, batch):
        gnn_emb = self.backbone.get_embedding(batch)
        fp = batch.fp.view(gnn_emb.size(0), -1)
        fp_emb = self.fp_encoder(fp)

        x = torch.cat([gnn_emb, fp_emb], dim=-1)
        x = self.shared(x)

        logits = {}
        for i, name in enumerate(CARDIAC_TARGETS):
            logits[name] = self.heads[i](x)

        return logits


def load_cardiac_data(data_dir: Path) -> pd.DataFrame:
    """Load all cardiac ion channel data."""
    print("Loading cardiac panel data...")

    all_data = []

    # hERG
    herg_path = data_dir / 'hERG_clean.csv'
    if herg_path.exists():
        df = pd.read_csv(herg_path)
        df['target'] = 'hERG'
        all_data.append(df[['smiles', 'active', 'target']])
        print(f"  hERG: {len(df)} molecules")

    # Nav1.5 and Cav1.2 from merged
    merged_path = data_dir / 'hERG_merged.csv'
    if merged_path.exists():
        df = pd.read_csv(merged_path)
        for target in ['Nav1.5', 'Cav1.2']:
            subset = df[df['source'] == f'ChEMBL_{target}']
            if len(subset) > 0:
                subset = subset[['smiles', 'active']].copy()
                subset['target'] = target
                all_data.append(subset)
                print(f"  {target}: {len(subset)} molecules")

    if not all_data:
        raise FileNotFoundError("No cardiac data found! Run pull_herg_extra.py first.")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined)}")
    print(f"Unique molecules: {combined['smiles'].nunique()}")

    return combined


def create_multitask_data(df: pd.DataFrame, featurizer: MoleculeGraphFeaturizer) -> List[Data]:
    """Create multi-task graphs."""
    pivot = df.pivot_table(
        index='smiles',
        columns='target',
        values='active',
        aggfunc='max'
    ).reset_index()

    print(f"Creating graphs for {len(pivot)} unique molecules...")

    graphs = []
    for _, row in tqdm(pivot.iterrows(), total=len(pivot), desc="Featurizing"):
        try:
            data = featurizer.featurize(row['smiles'], {'DAT': -1, 'NET': -1, 'SERT': -1})
            if data is None:
                continue

            fp = get_fingerprint(row['smiles'])
            if fp is None:
                continue

            labels = []
            mask = []
            for target in CARDIAC_TARGETS:
                if target in row and pd.notna(row[target]):
                    labels.append(int(row[target]))
                    mask.append(1)
                else:
                    labels.append(0)
                    mask.append(0)

            data.y = torch.tensor(labels, dtype=torch.long)
            data.mask = torch.tensor(mask, dtype=torch.float32)
            data.fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
            graphs.append(data)

        except Exception:
            continue

    print(f"Valid graphs: {len(graphs)}")
    return graphs


def compute_loss(outputs, batch, device):
    total_loss = 0
    n_tasks = 0

    for i, target in enumerate(CARDIAC_TARGETS):
        logits = outputs[target]
        labels = batch.y[:, i].to(device)
        mask = batch.mask[:, i].to(device)

        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        total_loss += loss
        n_tasks += 1

    return total_loss / max(n_tasks, 1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch)
        loss = compute_loss(outputs, batch, device)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_probs = {t: [] for t in CARDIAC_TARGETS}
    all_labels = {t: [] for t in CARDIAC_TARGETS}
    all_masks = {t: [] for t in CARDIAC_TARGETS}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        for i, target in enumerate(CARDIAC_TARGETS):
            probs = torch.softmax(outputs[target], dim=-1)[:, 1]
            all_probs[target].extend(probs.cpu().numpy())
            all_labels[target].extend(batch.y[:, i].cpu().numpy())
            all_masks[target].extend(batch.mask[:, i].cpu().numpy())

    metrics = {}
    aucs = []

    for target in CARDIAC_TARGETS:
        probs = np.array(all_probs[target])
        labels = np.array(all_labels[target])
        mask = np.array(all_masks[target]).astype(bool)

        if mask.sum() > 10 and len(np.unique(labels[mask])) > 1:
            auc = roc_auc_score(labels[mask], probs[mask])
            metrics[f'{target}_auc'] = auc
            aucs.append(auc)

    metrics['mean_auc'] = np.mean(aucs) if aucs else 0
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Cardiac Ion Channel Panel')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--data-dir', type=str, default='data/toxicity')
    parser.add_argument('--output-dir', type=str, default='models/cardiac')
    args = parser.parse_args()

    print("=" * 70)
    print("CARDIAC ION CHANNEL PANEL (CiPA)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Targets: {', '.join(CARDIAC_TARGETS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    df = load_cardiac_data(data_dir)

    # Create graphs
    featurizer = MoleculeGraphFeaturizer(use_3d=False)
    all_graphs = create_multitask_data(df, featurizer)

    # Split
    train_graphs, val_graphs = train_test_split(all_graphs, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(train_graphs)}, Val: {len(val_graphs)}")

    train_loader = PyGDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    # Model
    backbone = StereoGNN()
    model = CardiacPanelClassifier(backbone, n_targets=len(CARDIAC_TARGETS)).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training
    best_auc = 0
    patience = 15
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        scheduler.step(val_metrics['mean_auc'])

        mean_auc = val_metrics['mean_auc']

        if epoch % 5 == 0 or mean_auc > best_auc:
            auc_str = ' | '.join([f"{t}: {val_metrics.get(f'{t}_auc', 0):.3f}" for t in CARDIAC_TARGETS])
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Mean AUC: {mean_auc:.4f}")
            print(f"          | {auc_str}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / 'best_cardiac_model.pt')
            print(f"  -> Best model saved!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    best_ckpt = torch.load(output_dir / 'best_cardiac_model.pt', map_location=DEVICE, weights_only=False)
    print(f"Best Mean AUC: {best_ckpt['val_metrics']['mean_auc']:.4f}")
    for target in CARDIAC_TARGETS:
        print(f"  {target}: {best_ckpt['val_metrics'].get(f'{target}_auc', 0):.4f}")

    results = {
        'best_mean_auc': best_ckpt['val_metrics']['mean_auc'],
        'per_target': {t: best_ckpt['val_metrics'].get(f'{t}_auc', 0) for t in CARDIAC_TARGETS},
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
