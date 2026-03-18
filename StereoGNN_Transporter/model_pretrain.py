"""
StereoGNN Pretraining Model Architecture
=========================================

Two-stage architecture:
1. Pretrain on ALL transporter data (general substrate capability)
2. Fine-tune on monoamine-specific data (DAT/NET/SERT)

The key insight: transporter substrates share common features
(amphiphilic, cationic, specific molecular weight range).
Learning these general patterns helps monoamine-specific prediction.
"""

import math
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import (
    GATv2Conv, global_add_pool, MessagePassing,
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax

from config import CONFIG, ModelConfig


class StereoAwareNodeEncoder(nn.Module):
    """Encodes node features with explicit handling of stereochemistry."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stereo_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.stereo_dim = stereo_dim

        # Base atom features (first 78) + stereo (last 11)
        self.base_dim = input_dim - 11
        self.stereo_input_dim = 11

        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.stereo_encoder = nn.Sequential(
            nn.Linear(self.stereo_input_dim, stereo_dim),
            nn.LayerNorm(stereo_dim),
            nn.Tanh(),
            nn.Linear(stereo_dim, stereo_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + stereo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.stereo_gate = nn.Sequential(
            nn.Linear(self.stereo_input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        base_features = x[:, :self.base_dim]
        stereo_features = x[:, self.base_dim:]

        base_encoded = self.base_encoder(base_features)
        stereo_encoded = self.stereo_encoder(stereo_features)

        stereo_importance = self.stereo_gate(stereo_features)
        stereo_weighted = stereo_encoded * stereo_importance

        combined = torch.cat([base_encoded, stereo_weighted], dim=-1)
        return self.fusion(combined)


class StereoAwareEdgeEncoder(nn.Module):
    """Encodes edge features with stereo bond information."""

    def __init__(self, input_dim: int, hidden_dim: int, stereo_dim: int = 16):
        super().__init__()

        self.base_dim = input_dim - 7
        self.stereo_input_dim = 7

        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.stereo_encoder = nn.Sequential(
            nn.Linear(self.stereo_input_dim, stereo_dim),
            nn.LayerNorm(stereo_dim),
            nn.Tanh(),
        )

        self.fusion = nn.Linear(hidden_dim + stereo_dim, hidden_dim)

    def forward(self, edge_attr: Tensor) -> Tensor:
        base = edge_attr[:, :self.base_dim]
        stereo = edge_attr[:, self.base_dim:]

        base_enc = self.base_encoder(base)
        stereo_enc = self.stereo_encoder(stereo)

        return self.fusion(torch.cat([base_enc, stereo_enc], dim=-1))


class GATLayer(nn.Module):
    """GAT layer with edge features and residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv = GATv2Conv(
            in_dim, out_dim // heads,
            heads=heads, dropout=dropout,
            edge_dim=edge_dim, concat=True,
        )

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        residual = self.residual(x)
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class AttentionReadout(nn.Module):
    """Attention-based graph readout for interpretability."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads),
        )

        self.transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        attn = self.attention(x)
        attn = softmax(attn, batch, dim=0)

        x_transformed = self.transform(x)
        x_heads = x_transformed.view(-1, self.num_heads, self.head_dim)

        weighted = attn.unsqueeze(-1) * x_heads
        graph_emb = global_add_pool(weighted.view(-1, self.hidden_dim), batch)

        return graph_emb, attn


class TaskHead(nn.Module):
    """Task-specific prediction head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class StereoGNNBackbone(nn.Module):
    """
    Shared backbone for StereoGNN.

    This is the pretrained component that learns general
    transporter substrate representations.
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()

        self.config = config or CONFIG.model

        # FIXED: Match model.py dimensions for transfer learning compatibility
        # 75 base atom features (from config) + 11 stereo features = 86
        # Note: featurizer.py line 254 defaults to 78, but config says 75
        # If mismatch occurs, check config.py atom_feature_dim
        self.node_input_dim = 86
        self.edge_input_dim = 18

        self.node_encoder = StereoAwareNodeEncoder(
            input_dim=self.node_input_dim,
            hidden_dim=self.config.atom_hidden_dim,
            stereo_dim=self.config.stereo_feature_dim,
            dropout=self.config.dropout,
        )

        self.edge_encoder = StereoAwareEdgeEncoder(
            input_dim=self.edge_input_dim,
            hidden_dim=self.config.bond_hidden_dim,
        )

        self.gnn_layers = nn.ModuleList()
        for i in range(self.config.num_gnn_layers):
            layer = GATLayer(
                in_dim=self.config.atom_hidden_dim,
                out_dim=self.config.atom_hidden_dim,
                edge_dim=self.config.bond_hidden_dim,
                heads=self.config.num_attention_heads,
                dropout=self.config.attention_dropout,
            )
            self.gnn_layers.append(layer)

        self.readout = AttentionReadout(
            hidden_dim=self.config.atom_hidden_dim,
            num_heads=self.config.num_attention_heads,
        )

        self.shared_layer = nn.Sequential(
            nn.Linear(self.config.atom_hidden_dim, self.config.shared_hidden_dim),
            nn.LayerNorm(self.config.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        self.last_node_attention = None

    def forward(self, data: Data) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns graph embeddings and optionally node attention.
        """
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_attr)

        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, edge_attr)

        graph_emb, node_attn = self.readout(x, data.batch)
        self.last_node_attention = node_attn

        shared = self.shared_layer(graph_emb)

        return shared, node_attn


class StereoGNNPretrain(nn.Module):
    """
    StereoGNN for pretraining on ALL transporter data.

    Has heads for multiple transporter families to learn
    general substrate representations.
    """

    # All transporter targets for pretraining
    PRETRAIN_TARGETS = [
        # SLC6 - Monoamine transporters (PRIMARY)
        'DAT', 'NET', 'SERT',
        # SLC18 - Vesicular
        'VMAT1', 'VMAT2',
        # SLC6 - Amino acid
        'GAT1', 'GlyT1', 'GlyT2',
        # SLC22 - Organic cation/anion
        'OCT1', 'OCT2', 'OAT1', 'OAT3',
        # ABC - Efflux
        'P-gp', 'BCRP', 'MRP1',
        # OATP
        'OATP1B1', 'OATP1B3',
    ]

    def __init__(self, config: ModelConfig = None, targets: List[str] = None):
        super().__init__()

        self.config = config or CONFIG.model
        self.targets = targets or self.PRETRAIN_TARGETS

        # Shared backbone
        self.backbone = StereoGNNBackbone(config)

        # Task-specific heads for each transporter
        self.heads = nn.ModuleDict({
            target: TaskHead(
                self.config.shared_hidden_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            )
            for target in self.targets
        })

    def forward(
        self,
        data: Data,
        active_targets: List[str] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with optional target selection.

        Args:
            data: Batch of molecular graphs
            active_targets: Only compute these targets (for efficiency)
            return_attention: Include attention weights
        """
        targets = active_targets or self.targets

        shared, node_attn = self.backbone(data)

        output = {}
        for target in targets:
            if target in self.heads:
                output[target] = self.heads[target](shared)

        if return_attention:
            output['node_attention'] = node_attn
            output['graph_embedding'] = shared

        return output

    def get_embedding(self, data: Data) -> Tensor:
        """Get shared embedding for a molecule."""
        shared, _ = self.backbone(data)
        return shared


class StereoGNNFinetune(nn.Module):
    """
    StereoGNN for fine-tuning on monoamine transporters.

    Takes pretrained backbone and adds specialized heads
    for DAT/NET/SERT with higher capacity.
    """

    MONOAMINE_TARGETS = ['DAT', 'NET', 'SERT']

    def __init__(
        self,
        pretrained_backbone: StereoGNNBackbone = None,
        config: ModelConfig = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.config = config or CONFIG.model

        # Use pretrained or new backbone
        if pretrained_backbone is not None:
            self.backbone = pretrained_backbone
        else:
            self.backbone = StereoGNNBackbone(config)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Higher-capacity monoamine-specific heads
        self.heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(self.config.shared_hidden_dim, self.config.task_specific_dim),
                nn.LayerNorm(self.config.task_specific_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.task_specific_dim, self.config.task_specific_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.task_specific_dim, self.config.num_classes),
            )
            for target in self.MONOAMINE_TARGETS
        })

        # Cross-transporter attention for multi-task learning
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.shared_hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

    def forward(
        self,
        data: Data,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass for monoamine prediction."""
        shared, node_attn = self.backbone(data)

        output = {}
        for target in self.MONOAMINE_TARGETS:
            output[target] = self.heads[target](shared)

        if return_attention:
            output['node_attention'] = node_attn
            output['graph_embedding'] = shared

        return output

    def predict_with_uncertainty(
        self,
        data: Data,
        n_samples: int = 30,
    ) -> Dict[str, Dict[str, Tensor]]:
        """MC Dropout uncertainty estimation."""
        self.train()

        predictions = {task: [] for task in self.MONOAMINE_TARGETS}

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(data)
                for task in predictions:
                    probs = F.softmax(output[task], dim=-1)
                    predictions[task].append(probs)

        self.eval()

        results = {}
        for task in predictions:
            stacked = torch.stack(predictions[task], dim=0)
            results[task] = {
                'mean': stacked.mean(dim=0),
                'std': stacked.std(dim=0),
            }

        return results

    @classmethod
    def from_pretrained(
        cls,
        pretrain_model: StereoGNNPretrain,
        freeze_backbone: bool = False,
    ) -> 'StereoGNNFinetune':
        """Create fine-tuning model from pretrained model."""
        return cls(
            pretrained_backbone=pretrain_model.backbone,
            config=pretrain_model.config,
            freeze_backbone=freeze_backbone,
        )


class StereoGNNForAblation(StereoGNNFinetune):
    """
    Version WITHOUT stereochemistry features.
    For ablation studies.
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__(config=config)

        # Override with non-stereo encoders
        self.backbone.node_encoder = nn.Sequential(
            nn.Linear(78, self.config.atom_hidden_dim),
            nn.LayerNorm(self.config.atom_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.atom_hidden_dim, self.config.atom_hidden_dim),
        )

        self.backbone.edge_encoder = nn.Sequential(
            nn.Linear(11, self.config.bond_hidden_dim),
            nn.LayerNorm(self.config.bond_hidden_dim),
            nn.ReLU(),
        )

    def forward(self, data: Data, return_attention: bool = False) -> Dict[str, Tensor]:
        # Use only base features
        data = data.clone()
        data.x = data.x[:, :78]
        data.edge_attr = data.edge_attr[:, :11]

        return super().forward(data, return_attention)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained(checkpoint_path: str, config: ModelConfig = None) -> StereoGNNPretrain:
    """Load pretrained model from checkpoint."""
    model = StereoGNNPretrain(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def create_finetuning_model(
    pretrain_checkpoint: str = None,
    freeze_backbone: bool = False,
) -> StereoGNNFinetune:
    """Create fine-tuning model, optionally from pretrained."""
    if pretrain_checkpoint is not None:
        pretrain_model = load_pretrained(pretrain_checkpoint)
        return StereoGNNFinetune.from_pretrained(
            pretrain_model,
            freeze_backbone=freeze_backbone,
        )
    else:
        return StereoGNNFinetune()


if __name__ == "__main__":
    print("=" * 70)
    print("StereoGNN Pretraining Model Test")
    print("=" * 70)

    # Test pretraining model
    pretrain_model = StereoGNNPretrain()
    print(f"Pretrain model parameters: {count_parameters(pretrain_model):,}")
    print(f"Targets: {pretrain_model.targets}")

    # Test fine-tuning model
    finetune_model = StereoGNNFinetune()
    print(f"\nFinetune model parameters: {count_parameters(finetune_model):,}")

    # Test from pretrained
    finetune_from_pretrain = StereoGNNFinetune.from_pretrained(
        pretrain_model, freeze_backbone=True
    )
    trainable = count_parameters(finetune_from_pretrain)
    total = sum(p.numel() for p in finetune_from_pretrain.parameters())
    print(f"From pretrained (frozen backbone): {trainable:,} / {total:,} trainable")

    # Ablation model
    ablation_model = StereoGNNForAblation()
    print(f"\nAblation model parameters: {count_parameters(ablation_model):,}")

    print("\n" + "=" * 70)
    print("Model architecture ready for pretraining + fine-tuning")
    print("=" * 70)
