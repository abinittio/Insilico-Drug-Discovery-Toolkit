#!/usr/bin/env python
"""
StereoGNN Training Script - Full Pretrain + Finetune Pipeline
==============================================================

Usage:
    python run_training.py                    # Full pipeline (pretrain + finetune)
    python run_training.py --finetune-only    # Skip pretraining (use cached pretrain)
    python run_training.py --pretrain-only    # Only pretrain, no fine-tuning
    python run_training.py --test-run         # Quick test (2 epochs each phase)
    python run_training.py --cpu              # Force CPU training
    python run_training.py --small            # Use smaller model for CPU

For GPU training:
    CUDA_VISIBLE_DEVICES=0 python run_training.py

The Full Pipeline:
    Phase 1: PRETRAIN on ALL transporters (SLC6, SLC22, ABC, etc.)
             Learn general "substrate-ness" representations
    Phase 2: FINETUNE on monoamine transporters (DAT, NET, SERT)
             Specialize for substrate vs blocker distinction

CPU Training Notes:
    - Use --small flag for reduced model size (faster on CPU)
    - Use --batch-size 8 or 16 for CPU
    - Pretraining takes ~30-60 min on CPU (--small)
    - Fine-tuning takes ~10-15 min on CPU (--small)
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

# Local imports
from config import CONFIG, STEREOSELECTIVE_PAIRS
from featurizer import MoleculeGraphFeaturizer
from model_pretrain import (
    StereoGNNPretrain, StereoGNNFinetune, StereoGNNForAblation,
    StereoGNNBackbone, count_parameters
)
from losses import FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# SMALL MODEL FOR CPU TRAINING
# =============================================================================

class StereoGNNSmallPretrain(nn.Module):
    """Small pretraining model for CPU - learns general substrate features."""

    # All targets for pretraining
    PRETRAIN_TARGETS = [
        'DAT', 'NET', 'SERT',  # Monoamine (primary)
        'VMAT1', 'VMAT2',  # Vesicular
        'GAT1', 'GlyT1',  # Amino acid
        'OCT1', 'OCT2',  # Organic cation
        'P-gp', 'BCRP',  # Efflux
    ]

    def __init__(self, node_dim: int = 86, edge_dim: int = 18, targets: List[str] = None):
        super().__init__()

        self.targets = targets or self.PRETRAIN_TARGETS
        self.hidden_dim = 128
        self.num_layers = 2
        self.num_heads = 2
        self.dropout = 0.1

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = GATv2Conv(
                self.hidden_dim, self.hidden_dim // self.num_heads,
                heads=self.num_heads, dropout=self.dropout,
                edge_dim=64, concat=True,
            )
            self.gnn_layers.append(conv)

        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

        # Shared readout
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )

        # Task heads for each transporter
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(64, 3),  # inactive/blocker/substrate
            )
            for target in self.targets
        })

    def get_embedding(self, data) -> torch.Tensor:
        """Get graph-level embedding (for transfer)."""
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for i, conv in enumerate(self.gnn_layers):
            x_new = conv(x, data.edge_index, edge_attr)
            x_new = self.norms[i](x_new)
            x = F.relu(x_new) + x

        graph_emb = global_mean_pool(x, data.batch)
        return self.readout(graph_emb)

    def forward(self, data, active_targets: List[str] = None, return_attention: bool = False):
        targets = active_targets or self.targets
        graph_emb = self.get_embedding(data)

        output = {}
        for target in targets:
            if target in self.heads:
                output[target] = self.heads[target](graph_emb)

        if return_attention:
            output['graph_embedding'] = graph_emb

        return output


class StereoGNNSmallFinetune(nn.Module):
    """Small finetuning model for CPU - specialized for monoamines."""

    MONOAMINE_TARGETS = ['DAT', 'NET', 'SERT']

    def __init__(
        self,
        node_dim: int = 86,
        edge_dim: int = 18,
        pretrained_model: StereoGNNSmallPretrain = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.hidden_dim = 128
        self.num_layers = 2
        self.num_heads = 2
        self.dropout = 0.1

        if pretrained_model is not None:
            # Copy pretrained backbone weights
            self.node_encoder = pretrained_model.node_encoder
            self.edge_encoder = pretrained_model.edge_encoder
            self.gnn_layers = pretrained_model.gnn_layers
            self.norms = pretrained_model.norms
            self.readout = pretrained_model.readout

            if freeze_backbone:
                for param in self.node_encoder.parameters():
                    param.requires_grad = False
                for param in self.edge_encoder.parameters():
                    param.requires_grad = False
                for layer in self.gnn_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                for norm in self.norms:
                    for param in norm.parameters():
                        param.requires_grad = False
        else:
            # Fresh initialization
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
            )
            self.gnn_layers = nn.ModuleList()
            for _ in range(self.num_layers):
                conv = GATv2Conv(
                    self.hidden_dim, self.hidden_dim // self.num_heads,
                    heads=self.num_heads, dropout=self.dropout,
                    edge_dim=64, concat=True,
                )
                self.gnn_layers.append(conv)
            self.norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
            ])
            self.readout = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )

        # NEW specialized heads for monoamines (higher capacity)
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(self.hidden_dim, 96),
                nn.LayerNorm(96),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(48, 3),
            )
            for target in self.MONOAMINE_TARGETS
        })

    def forward(self, data, return_attention: bool = False):
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
            output[target] = self.heads[target](graph_emb)

        if return_attention:
            output['graph_embedding'] = graph_emb

        return output


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pretraining_data() -> pd.DataFrame:
    """Load ALL transporter data for pretraining."""
    logger.info("Loading pretraining data (all transporters)...")

    # Try to load cached pretraining data
    cache_file = Path("./data/pretrain_data.parquet")
    if cache_file.exists():
        logger.info(f"Loading cached pretraining data from {cache_file}")
        df = pd.read_parquet(cache_file)
        logger.info(f"  Loaded {len(df)} records, {df['smiles'].nunique()} unique compounds")
        return df

    # Generate pretraining data
    logger.info("Generating pretraining data...")

    # Try ChEMBL fetcher
    try:
        from data_all_transporters import create_pretraining_dataset
        general_df, monoamine_df = create_pretraining_dataset()

        if len(general_df) > 100:
            # Save cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            general_df.to_parquet(cache_file)
            return general_df
    except Exception as e:
        logger.warning(f"Could not fetch ChEMBL data: {e}")

    # Fallback: expand monoamine data with synthetic targets
    logger.info("Using fallback pretraining data generation...")
    return _generate_synthetic_pretrain_data()


def _generate_synthetic_pretrain_data() -> pd.DataFrame:
    """Generate synthetic pretraining data from monoamine compounds."""
    from data_comprehensive import ComprehensiveLiteratureData
    from data_sar_expansion import SARExpander, DecoyGenerator
    from data_augmentation import StereoisomerEnumerator
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem

    # Load monoamine data
    lit_df = ComprehensiveLiteratureData.get_all_data()

    expander = SARExpander()
    sar_df = expander.generate_all()

    decoy_gen = DecoyGenerator()
    decoy_df = decoy_gen.generate_decoys()

    monoamine_df = pd.concat([lit_df, sar_df, decoy_df], ignore_index=True)

    # Augment with stereoisomers
    enumerator = StereoisomerEnumerator()
    monoamine_df = enumerator.augment_dataset(monoamine_df)

    # Create pseudo-targets based on molecular properties
    # This simulates having data from multiple transporter families
    records = []

    for _, row in tqdm(monoamine_df.iterrows(), total=len(monoamine_df), desc="Generating pretrain data"):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Original monoamine target
        records.append({
            'smiles': smiles,
            'target': row['target'],
            'label': row['label'],
            'target_family': 'SLC6',
            'source': 'literature',
        })

        # Generate pseudo-labels for other transporter families based on properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        charge = Chem.GetFormalCharge(mol)

        # OCT substrates: cationic, moderate lipophilicity
        if charge > 0 or logp > 0:
            oct_label = 2 if logp < 3 and mw < 400 else (1 if logp < 4 else 0)
            records.append({
                'smiles': smiles,
                'target': 'OCT1',
                'label': oct_label,
                'target_family': 'SLC22',
                'source': 'synthetic',
            })

        # P-gp substrates: larger, lipophilic
        if mw > 300:
            pgp_label = 2 if logp > 2 and mw > 400 else (1 if logp > 1 else 0)
            records.append({
                'smiles': smiles,
                'target': 'P-gp',
                'label': pgp_label,
                'target_family': 'ABC',
                'source': 'synthetic',
            })

        # VMAT2: similar to DAT/NET (monoamines)
        if row['target'] in ['DAT', 'NET']:
            records.append({
                'smiles': smiles,
                'target': 'VMAT2',
                'label': row['label'],
                'target_family': 'SLC18',
                'source': 'synthetic',
            })

        # GAT1: small, polar amino acid analogs
        if tpsa > 40 and mw < 250:
            gat_label = 2 if tpsa > 60 else (1 if tpsa > 50 else 0)
            records.append({
                'smiles': smiles,
                'target': 'GAT1',
                'label': gat_label,
                'target_family': 'SLC6',
                'source': 'synthetic',
            })

    pretrain_df = pd.DataFrame(records)

    # Cache it
    cache_file = Path("./data/pretrain_data.parquet")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pretrain_df.to_parquet(cache_file)

    logger.info(f"Generated {len(pretrain_df)} pretraining records")
    logger.info(f"Targets: {pretrain_df['target'].unique().tolist()}")

    return pretrain_df


def load_finetuning_data() -> pd.DataFrame:
    """Load monoamine-specific data for fine-tuning."""
    logger.info("Loading fine-tuning data (monoamines only)...")

    from data_comprehensive import ComprehensiveLiteratureData
    from data_sar_expansion import SARExpander, DecoyGenerator
    from data_augmentation import StereoisomerEnumerator

    # Literature data
    lit_df = ComprehensiveLiteratureData.get_all_data()
    logger.info(f"  Literature: {lit_df['smiles'].nunique()} compounds")

    # SAR expansion
    expander = SARExpander()
    sar_df = expander.generate_all()
    logger.info(f"  SAR expanded: {sar_df['smiles'].nunique()} compounds")

    # Decoys
    decoy_gen = DecoyGenerator()
    decoy_df = decoy_gen.generate_decoys()
    logger.info(f"  Decoys: {decoy_df['smiles'].nunique()} compounds")

    # Combine
    combined = pd.concat([lit_df, sar_df, decoy_df], ignore_index=True)
    logger.info(f"  Combined: {combined['smiles'].nunique()} unique compounds")

    # Augment with stereoisomers
    enumerator = StereoisomerEnumerator()
    augmented = enumerator.augment_dataset(combined)
    logger.info(f"  After stereo augmentation: {augmented['smiles'].nunique()} compounds")

    return augmented


# =============================================================================
# GRAPH CREATION
# =============================================================================

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

        # Add labels as attributes
        setattr(graph, f'{target}_label', torch.tensor([label], dtype=torch.long))

        # Add -1 for other targets (not applicable)
        for other_target in targets:
            if other_target != target:
                if not hasattr(graph, f'{other_target}_label'):
                    setattr(graph, f'{other_target}_label', torch.tensor([-1], dtype=torch.long))

        graphs.append(graph)

    return graphs


def collate_fn_pretrain(data_list: List[Data]) -> Batch:
    """Custom collate for pretraining (all targets)."""
    batch = Batch.from_data_list(data_list)

    all_targets = ['DAT', 'NET', 'SERT', 'VMAT1', 'VMAT2', 'GAT1', 'GlyT1',
                   'OCT1', 'OCT2', 'P-gp', 'BCRP']

    for target in all_targets:
        labels = []
        for data in data_list:
            if hasattr(data, f'{target}_label'):
                labels.append(getattr(data, f'{target}_label'))
            else:
                labels.append(torch.tensor([-1], dtype=torch.long))
        setattr(batch, f'{target}_label', torch.cat(labels))

    return batch


def collate_fn_finetune(data_list: List[Data]) -> Batch:
    """Custom collate for fine-tuning (monoamines only)."""
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
    scheduler=None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch, active_targets=targets) if hasattr(model, 'targets') else model(batch)

        # Multi-task loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_tasks = 0

        for target in targets:
            if target not in outputs:
                continue
            attr_name = f'{target}_label'
            if not hasattr(batch, attr_name):
                continue
            labels = getattr(batch, attr_name)
            mask = labels >= 0

            if mask.sum() > 0:
                target_loss = loss_fn(outputs[target][mask], labels[mask])
                loss = loss + target_loss
                n_tasks += 1

        if n_tasks > 0:
            loss = loss / n_tasks
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
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
    """Evaluate model on a dataset."""
    model.eval()

    all_preds = {t: [] for t in targets}
    all_labels = {t: [] for t in targets}

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch, active_targets=targets) if hasattr(model, 'targets') else model(batch)

        for target in targets:
            if target not in outputs:
                continue
            attr_name = f'{target}_label'
            if not hasattr(batch, attr_name):
                continue
            labels = getattr(batch, attr_name)
            mask = labels >= 0

            if mask.sum() > 0:
                probs = F.softmax(outputs[target][mask], dim=-1)
                all_preds[target].append(probs.cpu().numpy())
                all_labels[target].append(labels[mask].cpu().numpy())

    # Compute metrics
    metrics = {}
    auc_scores = []
    pr_auc_scores = []

    for target in targets:
        if len(all_preds[target]) == 0:
            continue

        preds = np.vstack(all_preds[target])
        labels = np.concatenate(all_labels[target])

        # ROC-AUC
        try:
            auc_score = roc_auc_score(labels, preds, multi_class='ovr', average='macro')
            metrics[f'{target}_auc'] = auc_score
            auc_scores.append(auc_score)
        except Exception:
            pass

        # PR-AUC for substrate class
        try:
            binary_labels = (labels == 2).astype(int)
            if binary_labels.sum() > 0:
                substrate_probs = preds[:, 2]
                precision, recall, _ = precision_recall_curve(binary_labels, substrate_probs)
                pr_auc = auc(recall, precision)
                metrics[f'{target}_pr_auc'] = pr_auc
                pr_auc_scores.append(pr_auc)
        except Exception:
            pass

    if auc_scores:
        metrics['mean_auc'] = np.mean(auc_scores)
    if pr_auc_scores:
        metrics['mean_pr_auc'] = np.mean(pr_auc_scores)

    return metrics


def evaluate_stereo_sensitivity(
    model: nn.Module,
    featurizer: MoleculeGraphFeaturizer,
    device: torch.device,
) -> float:
    """Test on known stereoselective pairs."""
    model.eval()

    correct = 0
    total = 0

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

            d_substrate_prob = F.softmax(d_out[target], dim=-1)[0, 2].item()
            l_substrate_prob = F.softmax(l_out[target], dim=-1)[0, 2].item()

            if d_activity > l_activity:
                if d_substrate_prob > l_substrate_prob:
                    correct += 1
            else:
                if l_substrate_prob > d_substrate_prob:
                    correct += 1

            total += 1
            logger.info(f"  {pair.get('name', 'Pair')}: d={d_substrate_prob:.3f} l={l_substrate_prob:.3f}")

    return correct / total if total > 0 else 0.0


# =============================================================================
# PRETRAINING PHASE
# =============================================================================

def pretrain(
    args,
    device: torch.device,
    output_dir: Path,
    featurizer: MoleculeGraphFeaturizer,
    node_dim: int,
    edge_dim: int,
) -> nn.Module:
    """Phase 1: Pretrain on all transporter data."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: PRETRAINING ON ALL TRANSPORTERS")
    logger.info("="*70)

    # Check for cached pretrained model
    pretrain_checkpoint = output_dir / 'pretrain_model.pt'
    if pretrain_checkpoint.exists() and args.finetune_only:
        logger.info(f"Loading cached pretrained model from {pretrain_checkpoint}")
        if args.small:
            model = StereoGNNSmallPretrain(node_dim=node_dim, edge_dim=edge_dim)
        else:
            model = StereoGNNPretrain()
        checkpoint = torch.load(pretrain_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

    # Load pretraining data
    pretrain_df = load_pretraining_data()

    # Determine available targets
    available_targets = pretrain_df['target'].unique().tolist()
    logger.info(f"Available targets: {available_targets}")

    # Create graphs
    logger.info("Creating molecular graphs for pretraining...")
    graphs = create_graph_dataset(pretrain_df, featurizer, available_targets)
    logger.info(f"Created {len(graphs)} graphs")

    if len(graphs) < 100:
        logger.warning("Not enough pretraining data - skipping pretrain phase")
        return None

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    n_train = int(0.85 * len(graphs))
    n_val = int(0.10 * len(graphs))

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train+n_val]]

    logger.info(f"Pretrain split: Train={len(train_graphs)}, Val={len(val_graphs)}")

    # Create loaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pretrain)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pretrain)

    # Create model
    if args.small:
        model = StereoGNNSmallPretrain(node_dim=node_dim, edge_dim=edge_dim, targets=available_targets)
    else:
        model = StereoGNNPretrain(targets=available_targets)

    model.to(device)
    logger.info(f"Pretrain model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr * 2, weight_decay=1e-5)  # Higher LR for pretrain
    loss_fn = FocalLoss(gamma=2.0)

    # Training
    pretrain_epochs = args.pretrain_epochs
    best_auc = 0.0

    for epoch in range(pretrain_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, available_targets)
        val_metrics = evaluate(model, val_loader, device, available_targets)

        mean_auc = val_metrics.get('mean_auc', 0)

        logger.info(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}: "
                   f"loss={train_loss:.4f}, val_auc={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
            }, pretrain_checkpoint)

    # Load best
    checkpoint = torch.load(pretrain_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Pretraining complete! Best val AUC: {best_auc:.4f}")

    return model


# =============================================================================
# FINE-TUNING PHASE
# =============================================================================

def finetune(
    args,
    device: torch.device,
    output_dir: Path,
    featurizer: MoleculeGraphFeaturizer,
    pretrained_model: nn.Module,
    node_dim: int,
    edge_dim: int,
) -> Tuple[nn.Module, Dict]:
    """Phase 2: Fine-tune on monoamine data."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: FINE-TUNING ON MONOAMINE TRANSPORTERS")
    logger.info("="*70)

    # Load fine-tuning data
    finetune_df = load_finetuning_data()

    # Create graphs
    logger.info("Creating molecular graphs for fine-tuning...")
    monoamine_targets = ['DAT', 'NET', 'SERT']
    graphs = create_graph_dataset(finetune_df, featurizer, monoamine_targets)
    logger.info(f"Created {len(graphs)} graphs")

    # Split
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    n_train = int(0.8 * len(graphs))
    n_val = int(0.1 * len(graphs))

    train_graphs = [graphs[i] for i in indices[:n_train]]
    val_graphs = [graphs[i] for i in indices[n_train:n_train+n_val]]
    test_graphs = [graphs[i] for i in indices[n_train+n_val:]]

    logger.info(f"Fine-tune split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # Create loaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_finetune)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_finetune)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_finetune)

    # Create model from pretrained
    if args.small:
        model = StereoGNNSmallFinetune(
            node_dim=node_dim,
            edge_dim=edge_dim,
            pretrained_model=pretrained_model if isinstance(pretrained_model, StereoGNNSmallPretrain) else None,
            freeze_backbone=False,  # Don't freeze - let it adapt
        )
    else:
        if pretrained_model is not None:
            model = StereoGNNFinetune.from_pretrained(pretrained_model, freeze_backbone=False)
        else:
            model = StereoGNNFinetune()

    model.to(device)
    logger.info(f"Fine-tune model parameters: {count_parameters(model):,}")

    # Optimizer with lower LR for fine-tuning
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = FocalLoss(gamma=2.0)

    # Training
    num_epochs = args.epochs
    best_auc = 0.0
    best_metrics = {}
    patience = 15
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, monoamine_targets)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, monoamine_targets)

        mean_auc = val_metrics.get('mean_auc', 0)

        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"loss={train_loss:.4f}, val_auc={mean_auc:.4f}, "
                   f"DAT={val_metrics.get('DAT_auc', 0):.3f}, "
                   f"NET={val_metrics.get('NET_auc', 0):.3f}, "
                   f"SERT={val_metrics.get('SERT_auc', 0):.3f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_metrics = val_metrics.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_auc': best_auc,
                'metrics': best_metrics,
            }, output_dir / 'best_model.pt')
            logger.info(f"  -> New best model saved!")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"Early stopping after {patience} epochs without improvement")
            break

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION")
    logger.info("="*70)

    test_metrics = evaluate(model, test_loader, device, monoamine_targets)
    logger.info(f"Test Metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Stereo sensitivity
    logger.info("\nStereo Sensitivity Test:")
    stereo_sens = evaluate_stereo_sensitivity(model, featurizer, device)
    logger.info(f"Stereo sensitivity: {stereo_sens:.2%}")

    test_metrics['stereo_sensitivity'] = stereo_sens

    return model, test_metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(args):
    """Main training pipeline: Pretrain + Finetune."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.cpu:
        device = torch.device('cpu')
        logger.info("Forcing CPU training")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        import multiprocessing
        logger.info(f"CPU cores: {multiprocessing.cpu_count()}")
        if not args.small:
            logger.warning("Consider using --small for CPU training")

    # Create featurizer
    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    # Get feature dimensions from a sample molecule
    from rdkit import Chem
    sample_mol = Chem.MolFromSmiles("CC(N)Cc1ccccc1")  # Amphetamine
    sample_graph = featurizer.featurize(Chem.MolToSmiles(sample_mol))
    node_dim = sample_graph.x.shape[1]
    edge_dim = sample_graph.edge_attr.shape[1]
    logger.info(f"Feature dimensions: node={node_dim}, edge={edge_dim}")

    # Phase 1: Pretrain
    pretrained_model = None
    pretrain_checkpoint = output_dir / 'pretrain_model.pt'

    if args.finetune_only and pretrain_checkpoint.exists():
        # Load existing pretrained model
        logger.info(f"Loading pretrained model from {pretrain_checkpoint}")
        checkpoint = torch.load(pretrain_checkpoint, map_location=device, weights_only=False)

        # Detect which targets were used in pretraining from the checkpoint
        pretrain_targets = []
        for key in checkpoint['model_state_dict'].keys():
            if key.startswith('heads.') and '.0.weight' in key:
                target = key.split('.')[1]
                pretrain_targets.append(target)
        logger.info(f"Pretrained targets: {pretrain_targets}")

        if args.small:
            pretrained_model = StereoGNNSmallPretrain(node_dim=node_dim, edge_dim=edge_dim, targets=pretrain_targets)
        else:
            pretrained_model = StereoGNNPretrain(targets=pretrain_targets)
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_model.to(device)
        logger.info(f"Loaded pretrained backbone (val_auc: {checkpoint.get('best_auc', 'N/A')})")
    elif not args.finetune_only:
        # Run pretraining
        pretrained_model = pretrain(args, device, output_dir, featurizer, node_dim, edge_dim)
    else:
        logger.warning("No pretrained model found and --finetune-only specified. Training from scratch.")

    if args.pretrain_only:
        logger.info("Pretraining complete. Exiting (--pretrain-only flag)")
        return

    # Phase 2: Fine-tune
    model, test_metrics = finetune(args, device, output_dir, featurizer, pretrained_model, node_dim, edge_dim)

    # Check success criteria
    logger.info("\n" + "="*70)
    logger.info("SUCCESS CRITERIA")
    logger.info("="*70)

    criteria = [
        ("Overall ROC-AUC >= 0.85", test_metrics.get('mean_auc', 0) >= 0.85),
        ("Substrate PR-AUC >= 0.65", test_metrics.get('mean_pr_auc', 0) >= 0.65),
        ("Stereo sensitivity >= 80%", test_metrics.get('stereo_sensitivity', 0) >= 0.80),
    ]

    all_pass = True
    for name, passed in criteria:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    # Save results
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'criteria_passed': all_pass,
        'pretrained': pretrained_model is not None,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("="*70)
    logger.info("TRAINING COMPLETE" + (" - ALL CRITERIA PASSED!" if all_pass else ""))
    logger.info("="*70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train StereoGNN with Pretrain + Finetune")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Fine-tuning epochs")
    parser.add_argument("--pretrain-epochs", type=int, default=30,
                       help="Pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--test-run", action="store_true",
                       help="Quick test (2 epochs each phase)")
    parser.add_argument("--pretrain-only", action="store_true",
                       help="Only run pretraining")
    parser.add_argument("--finetune-only", action="store_true",
                       help="Skip pretraining (use cached)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU training")
    parser.add_argument("--small", action="store_true",
                       help="Use smaller model for CPU")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")

    args = parser.parse_args()

    if args.test_run:
        args.epochs = 2
        args.pretrain_epochs = 2
        logger.info("TEST RUN MODE: 2 epochs each phase")

    main(args)
