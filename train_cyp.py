"""
CYP450 Multi-Task Model
========================
Predicts inhibition of major drug-metabolizing enzymes.
Critical for drug-drug interaction and clearance prediction.

Targets: CYP3A4, CYP2D6, CYP2C9, CYP2C19, CYP1A2
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from model import StereoGNN, count_parameters
from featurizer import MoleculeGraphFeaturizer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CYP_TARGETS = ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4']
FP_DIM = 1024 + 8  # Morgan bits + descriptors


def get_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """Generate Morgan fingerprint + key descriptors."""
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


class CYPClassifier(nn.Module):
    """Multi-task CYP inhibition classifier with FP fusion."""
    def __init__(self, backbone, hidden_dim=128, fp_dim=FP_DIM, n_targets=5, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.n_targets = n_targets

        # Fingerprint encoder
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fused dimension
        fused_dim = hidden_dim + hidden_dim

        # Shared layer after fusion
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Deeper per-target heads with LayerNorm
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, 2),
            ) for _ in range(n_targets)
        ])

    def forward(self, batch):
        # GNN embedding
        gnn_emb = self.backbone.get_embedding(batch)

        # FP embedding
        fp = batch.fp.view(gnn_emb.size(0), -1)
        fp_emb = self.fp_encoder(fp)

        # Fuse and shared
        x = torch.cat([gnn_emb, fp_emb], dim=-1)
        x = self.shared(x)

        logits = {}
        for i, name in enumerate(CYP_TARGETS):
            logits[name] = self.heads[i](x)

        return logits


def load_cyp_data(data_dir: Path) -> pd.DataFrame:
    """Load all CYP datasets and merge."""
    print("Loading CYP data...")

    all_data = []
    for cyp in CYP_TARGETS:
        path = data_dir / f'{cyp}_chembl.csv'
        if path.exists():
            df = pd.read_csv(path)
            df['target'] = cyp
            all_data.append(df)
            print(f"  {cyp}: {len(df)} molecules, {df['active'].sum()} active")
        else:
            print(f"  {cyp}: NOT FOUND")

    if not all_data:
        raise FileNotFoundError("No CYP data found!")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined)}")
    print(f"Unique molecules: {combined['smiles'].nunique()}")

    return combined


def create_multitask_data(
    df: pd.DataFrame,
    featurizer: MoleculeGraphFeaturizer,
) -> List[Data]:
    """Create multi-task labeled graphs."""
    # Pivot to get one row per molecule with all CYP labels
    pivot = df.pivot_table(
        index='smiles',
        columns='target',
        values='active',
        aggfunc='max'  # If any assay says active, it's active
    ).reset_index()

    print(f"Creating graphs for {len(pivot)} unique molecules...")

    graphs = []
    for _, row in tqdm(pivot.iterrows(), total=len(pivot), desc="Featurizing"):
        try:
            data = featurizer.featurize(row['smiles'], {'DAT': -1, 'NET': -1, 'SERT': -1})
            if data is None:
                continue

            # Fingerprint
            fp = get_fingerprint(row['smiles'])
            if fp is None:
                continue

            # Multi-task labels: -1 = missing, 0 = inactive, 1 = active
            labels = []
            mask = []
            for cyp in CYP_TARGETS:
                if cyp in row and pd.notna(row[cyp]):
                    labels.append(int(row[cyp]))
                    mask.append(1)
                else:
                    labels.append(0)
                    mask.append(0)

            data.y = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
            data.mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            data.fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)
            graphs.append(data)

        except Exception:
            continue

    print(f"Valid graphs: {len(graphs)}")
    return graphs


# Class weights per CYP (based on active ratios)
CYP_CLASS_WEIGHTS = {
    'CYP1A2': 0.7,   # 70% active - downweight
    'CYP2C9': 1.3,   # 36% active - upweight
    'CYP2C19': 1.2,  # 30% active - upweight
    'CYP2D6': 1.3,   # 32% active - upweight
    'CYP3A4': 1.1,   # 40% active - slight upweight
}


def compute_loss(outputs: Dict[str, torch.Tensor], batch, device) -> torch.Tensor:
    """Compute masked multi-task loss with class weighting."""
    total_loss = 0
    n_tasks = 0

    # Reshape batched tensors
    batch_size = outputs[CYP_TARGETS[0]].size(0)
    y = batch.y.view(batch_size, -1)
    mask = batch.mask.view(batch_size, -1)

    for i, cyp in enumerate(CYP_TARGETS):
        logits = outputs[cyp]
        labels = y[:, i].to(device)
        m = mask[:, i].to(device)

        if m.sum() == 0:
            continue

        # Class-weighted cross-entropy
        weight = torch.ones_like(labels, dtype=torch.float32)
        weight[labels == 1] = CYP_CLASS_WEIGHTS[cyp]

        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss * m * weight).sum() / m.sum()
        total_loss += loss
        n_tasks += 1

    return total_loss / max(n_tasks, 1)


def train_epoch(model, loader, optimizer, device, scheduler=None):
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
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()

    all_probs = {cyp: [] for cyp in CYP_TARGETS}
    all_labels = {cyp: [] for cyp in CYP_TARGETS}
    all_masks = {cyp: [] for cyp in CYP_TARGETS}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        batch_size = outputs[CYP_TARGETS[0]].size(0)
        y = batch.y.view(batch_size, -1)
        mask = batch.mask.view(batch_size, -1)

        for i, cyp in enumerate(CYP_TARGETS):
            probs = torch.softmax(outputs[cyp], dim=-1)[:, 1]
            all_probs[cyp].extend(probs.cpu().numpy())
            all_labels[cyp].extend(y[:, i].cpu().numpy())
            all_masks[cyp].extend(mask[:, i].cpu().numpy())

    metrics = {}
    aucs = []

    for cyp in CYP_TARGETS:
        probs = np.array(all_probs[cyp])
        labels = np.array(all_labels[cyp])
        mask = np.array(all_masks[cyp]).astype(bool)

        if mask.sum() > 10 and len(np.unique(labels[mask])) > 1:
            auc = roc_auc_score(labels[mask], probs[mask])
            metrics[f'{cyp}_auc'] = auc
            aucs.append(auc)

    metrics['mean_auc'] = np.mean(aucs) if aucs else 0
    return metrics


def main():
    parser = argparse.ArgumentParser(description='CYP450 Multi-Task Model')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/toxicity', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='models/cyp', help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("CYP450 MULTI-TASK MODEL")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Targets: {', '.join(CYP_TARGETS)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    df = load_cyp_data(data_dir)

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
    model = CYPClassifier(backbone, n_targets=len(CYP_TARGETS)).to(DEVICE)
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
            auc_str = ' | '.join([f"{c}: {val_metrics.get(f'{c}_auc', 0):.3f}" for c in CYP_TARGETS])
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Mean AUC: {mean_auc:.4f}")
            print(f"          | {auc_str}")

        # Checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / f'checkpoint_epoch{epoch}.pt')

        if mean_auc > best_auc:
            best_auc = mean_auc
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / 'best_cyp_model.pt')
            print(f"  -> Best model saved!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    best_ckpt = torch.load(output_dir / 'best_cyp_model.pt', map_location=DEVICE, weights_only=False)
    print(f"Best Mean AUC: {best_ckpt['val_metrics']['mean_auc']:.4f}")
    for cyp in CYP_TARGETS:
        print(f"  {cyp}: {best_ckpt['val_metrics'].get(f'{cyp}_auc', 0):.4f}")

    # Save results
    results = {
        'best_mean_auc': best_ckpt['val_metrics']['mean_auc'],
        'per_target': {cyp: best_ckpt['val_metrics'].get(f'{cyp}_auc', 0) for cyp in CYP_TARGETS},
        'n_molecules': len(all_graphs),
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
