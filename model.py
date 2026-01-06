"""
StereoGNN Model Architecture
=============================

A stereochemistry-aware graph neural network for monoamine transporter substrate prediction.

Key innovations:
1. Explicit stereo feature encoding in node/edge features
2. Chirality-aware message passing
3. 3D coordinate-enhanced convolutions (optional)
4. Multi-task output heads for DAT/NET/SERT
5. MC Dropout for uncertainty quantification

Architecture:
- Node embedding with stereo features
- Edge embedding with stereo bond features
- Stacked GAT layers with attention for interpretability
- Attention-based graph readout
- Shared representation layer
- Task-specific prediction heads
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import (
    GATConv, GATv2Conv, GINConv, TransformerConv,
    global_mean_pool, global_add_pool, global_max_pool,
    Set2Set, MessagePassing,
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax

from config import CONFIG, ModelConfig


class StereoAwareNodeEncoder(nn.Module):
    """
    Encodes node features with explicit handling of stereochemistry.

    The key insight is that chiral atoms need special attention -
    their R/S configuration fundamentally changes binding behavior.
    """

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

        # Base atom features encoder
        # Split the input - first part is standard, last 11 are stereo
        self.base_dim = input_dim - 11  # 11 stereo features (9 chiral tags + R/S + is_center)
        self.stereo_input_dim = 11

        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stereo-specific encoder with dedicated capacity
        self.stereo_encoder = nn.Sequential(
            nn.Linear(self.stereo_input_dim, stereo_dim),
            nn.LayerNorm(stereo_dim),
            nn.Tanh(),  # Tanh because stereo is symmetric (+/-)
            nn.Linear(stereo_dim, stereo_dim),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + stereo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Learnable stereo importance
        self.stereo_gate = nn.Sequential(
            nn.Linear(self.stereo_input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Node features [N, input_dim]

        Returns:
            Encoded features [N, hidden_dim]
        """
        # Split features
        base_features = x[:, :self.base_dim]
        stereo_features = x[:, self.base_dim:]

        # Encode separately
        base_encoded = self.base_encoder(base_features)
        stereo_encoded = self.stereo_encoder(stereo_features)

        # Gate: learn how much to weight stereo based on presence
        stereo_importance = self.stereo_gate(stereo_features)

        # Weight stereo encoding
        stereo_weighted = stereo_encoded * stereo_importance

        # Fuse
        combined = torch.cat([base_encoded, stereo_weighted], dim=-1)
        output = self.fusion(combined)

        return output


class StereoAwareEdgeEncoder(nn.Module):
    """Encodes edge features with stereo bond information."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        stereo_dim: int = 16,
    ):
        super().__init__()

        # Base bond features (first 11) + stereo (last 7)
        self.base_dim = input_dim - 7  # 7 stereo features (6 E/Z + is_stereo)
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


class ChiralMessagePassing(MessagePassing):
    """
    Message passing layer that is aware of chirality.

    When passing messages around a chiral center, we want the model
    to learn that the ORDER of neighbors matters (clockwise vs counter-clockwise).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads

        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)

        # Edge feature integration
        self.edge_proj = nn.Linear(edge_dim, out_channels)

        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        # Compute attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Edge features
        edge_emb = self.edge_proj(edge_attr)

        # Propagate
        out = self.propagate(
            edge_index, q=q, k=k, v=v,
            edge_emb=edge_emb, size=None,
        )

        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual
        return self.layer_norm(x + out)

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        edge_emb: Tensor,
        index: Tensor,
    ) -> Tensor:
        # Attention scores
        # Include edge features in attention computation
        attn = (q_i * (k_j + edge_emb)).sum(dim=-1) / math.sqrt(self.out_channels)

        # Softmax over neighbors
        attn = softmax(attn, index)
        attn = self.dropout(attn)

        # Weighted values
        return attn.unsqueeze(-1) * (v_j + edge_emb)


class GATLayer(nn.Module):
    """GAT layer with edge features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()

        self.conv = GATv2Conv(
            in_dim,
            out_dim // heads if concat else out_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat,
        )

        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        residual = self.residual(x)

        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        return out + residual


class AttentionReadout(nn.Module):
    """
    Attention-based graph readout.

    Learns which nodes are most important for the prediction.
    Critical for interpretability.
    """

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
        """
        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment [N]

        Returns:
            graph_emb: Graph embeddings [B, hidden_dim]
            attention_weights: Per-node attention [N, num_heads]
        """
        # Compute attention weights
        attn = self.attention(x)  # [N, num_heads]
        attn = softmax(attn, batch, dim=0)  # Softmax per graph

        # Transform features
        x_transformed = self.transform(x)  # [N, hidden_dim]

        # Reshape for multi-head attention
        x_heads = x_transformed.view(-1, self.num_heads, self.head_dim)

        # Weighted sum
        weighted = attn.unsqueeze(-1) * x_heads  # [N, heads, head_dim]

        # Pool per graph
        # First, sum within each graph
        graph_emb = global_add_pool(weighted.view(-1, self.hidden_dim), batch)

        return graph_emb, attn


class MultiTaskHead(nn.Module):
    """
    Task-specific prediction head for a single transporter.

    Outputs:
    - Class probabilities (substrate, blocker, inactive)
    - Uncertainty estimate via MC Dropout
    """

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
            nn.Dropout(dropout),  # MC Dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # MC Dropout
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class StereoGNN(nn.Module):
    """
    Complete Stereochemistry-aware GNN for transporter substrate prediction.

    Multi-task learning for DAT, NET, SERT with shared representation.
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()

        self.config = config or CONFIG.model

        # Get feature dimensions from config or featurizer
        # Actual dimensions from featurizer.py:
        # - Base atom features: 75 (one-hot encoded atomic properties)
        # - Stereo features: 11 (9 chiral tags + 1 R/S config + 1 is_stereocenter)
        # - Total node features: 86
        # - Edge features: 18 (11 bond features + 6 stereo bond + 1 is_stereo_bond)
        self.node_input_dim = 86  # From featurizer: 75 + 11
        self.edge_input_dim = 18  # From featurizer: 11 + 7

        # Node encoder (stereo-aware)
        self.node_encoder = StereoAwareNodeEncoder(
            input_dim=self.node_input_dim,
            hidden_dim=self.config.atom_hidden_dim,
            stereo_dim=self.config.stereo_feature_dim,
            dropout=self.config.dropout,
        )

        # Edge encoder (stereo-aware)
        self.edge_encoder = StereoAwareEdgeEncoder(
            input_dim=self.edge_input_dim,
            hidden_dim=self.config.bond_hidden_dim,
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(self.config.num_gnn_layers):
            in_dim = self.config.atom_hidden_dim
            out_dim = self.config.atom_hidden_dim

            layer = GATLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                edge_dim=self.config.bond_hidden_dim,
                heads=self.config.num_attention_heads,
                dropout=self.config.attention_dropout,
            )
            self.gnn_layers.append(layer)

        # Readout
        self.readout = AttentionReadout(
            hidden_dim=self.config.atom_hidden_dim,
            num_heads=self.config.num_attention_heads,
        )

        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.config.atom_hidden_dim, self.config.shared_hidden_dim),
            nn.LayerNorm(self.config.shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict({
            'DAT': MultiTaskHead(
                self.config.shared_hidden_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            ),
            'NET': MultiTaskHead(
                self.config.shared_hidden_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            ),
            'SERT': MultiTaskHead(
                self.config.shared_hidden_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            ),
        })

        # Store attention weights for interpretability
        self.last_node_attention = None
        self.last_layer_attention = None

    def forward(
        self,
        data: Data,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data object
            return_attention: Whether to return attention weights

        Returns:
            Dict with 'DAT', 'NET', 'SERT' logits and optionally 'attention'
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Encode nodes
        x = self.node_encoder(x)

        # Encode edges
        edge_attr = self.edge_encoder(edge_attr)

        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Readout
        graph_emb, node_attn = self.readout(x, batch)

        # Store for interpretability
        self.last_node_attention = node_attn

        # Shared representation
        shared = self.shared_layer(graph_emb)

        # Task-specific predictions
        output = {
            'DAT': self.heads['DAT'](shared),
            'NET': self.heads['NET'](shared),
            'SERT': self.heads['SERT'](shared),
        }

        if return_attention:
            output['node_attention'] = node_attn
            output['graph_embedding'] = graph_emb

        return output

    def predict_with_uncertainty(
        self,
        data: Data,
        n_samples: int = 30,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Make predictions with MC Dropout uncertainty estimation.

        Returns mean predictions and uncertainty (std) for each task.
        """
        self.train()  # Enable dropout

        predictions = {task: [] for task in ['DAT', 'NET', 'SERT']}

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(data)
                for task in predictions:
                    probs = F.softmax(output[task], dim=-1)
                    predictions[task].append(probs)

        self.eval()

        results = {}
        for task in predictions:
            stacked = torch.stack(predictions[task], dim=0)  # [n_samples, batch, classes]
            results[task] = {
                'mean': stacked.mean(dim=0),
                'std': stacked.std(dim=0),
                'predictions': stacked,
            }

        return results

    def get_embedding(self, data: Data) -> Tensor:
        """Get the graph-level embedding for a molecule."""
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        graph_emb, _ = self.readout(x, batch)

        return graph_emb


class StereoGNNForAblation(StereoGNN):
    """
    Version of StereoGNN WITHOUT stereochemistry features.

    Used for ablation studies to prove stereo features matter.
    """

    def __init__(self, config: ModelConfig = None):
        # Create a modified config without stereo features
        super().__init__(config)

        # Override the node encoder to ignore stereo features
        self.node_input_dim = 75  # Only base features (actual from featurizer)

        # Simple encoder without stereo awareness
        self.node_encoder = nn.Sequential(
            nn.Linear(75, self.config.atom_hidden_dim),
            nn.LayerNorm(self.config.atom_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.atom_hidden_dim, self.config.atom_hidden_dim),
        )

        # Edge encoder without stereo
        self.edge_input_dim = 11  # Only base bond features
        self.edge_encoder = nn.Sequential(
            nn.Linear(11, self.config.bond_hidden_dim),
            nn.LayerNorm(self.config.bond_hidden_dim),
            nn.ReLU(),
        )

    def forward(self, data: Data, return_attention: bool = False) -> Dict[str, Tensor]:
        """Forward pass with stereo features removed."""
        # Take only non-stereo features
        x = data.x[:, :75]  # Remove stereo features (base features = 75)
        edge_attr = data.edge_attr[:, :11]  # Remove stereo bond features
        edge_index = data.edge_index
        batch = data.batch

        # Encode
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Readout
        graph_emb, node_attn = self.readout(x, batch)

        # Shared representation
        shared = self.shared_layer(graph_emb)

        # Task-specific predictions
        output = {
            'DAT': self.heads['DAT'](shared),
            'NET': self.heads['NET'](shared),
            'SERT': self.heads['SERT'](shared),
        }

        if return_attention:
            output['node_attention'] = node_attn
            output['graph_embedding'] = graph_emb

        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# KINETIC EXTENSION - Mechanistic parameter prediction
# =============================================================================

class KineticHead(nn.Module):
    """
    Kinetic parameter prediction head for one transporter.

    Outputs (all continuous + one classification):
    1. pKi: Binding affinity (-log10 Ki), range ~3-12
    2. pIC50: Functional potency (-log10 IC50), range ~3-12
    3. Interaction mode: 4-class (substrate, competitive, non-competitive, partial)
    4. Kinetic bias: Uptake/blockade ratio, range 0-1
    5. Heteroscedastic uncertainty estimates for each continuous output

    Design rationale:
    - Each regression head outputs [mean, log_var] for heteroscedastic uncertainty
    - GELU activation often better for regression tasks
    - Separate encoder allows kinetic-specific feature extraction
    """

    # Interaction mode encoding
    MODE_SUBSTRATE = 0
    MODE_COMPETITIVE = 1
    MODE_NONCOMPETITIVE = 2
    MODE_PARTIAL = 3
    MODE_NAMES = ['substrate', 'competitive_inhibitor', 'non_competitive_inhibitor', 'partial_substrate']

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared kinetic feature extractor
        self.kinetic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # === Regression heads with heteroscedastic uncertainty ===

        # Binding affinity (pKi): outputs mean + log_variance
        self.pki_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # [mean, log_var]
        )

        # Functional potency (pIC50): outputs mean + log_variance
        self.pic50_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # [mean, log_var]
        )

        # Kinetic bias (uptake preference 0-1): outputs mean + log_variance
        self.kinetic_bias_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # [mean, log_var]
        )

        # === Classification head ===

        # Interaction mode: 4 classes
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),  # substrate, competitive, non-competitive, partial
        )

        # Initialize output layers with small weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for regression heads."""
        for head in [self.pki_head, self.pic50_head, self.kinetic_bias_head]:
            # Initialize final layer with small weights
            nn.init.xavier_uniform_(head[-1].weight, gain=0.1)
            nn.init.zeros_(head[-1].bias)
            # Initialize log_var bias to small negative value (low initial uncertainty)
            head[-1].bias.data[1] = -2.0

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass for kinetic parameter prediction.

        Args:
            x: Shared representation [batch, input_dim]

        Returns:
            Dict with all kinetic parameters and uncertainties
        """
        h = self.kinetic_encoder(x)

        # Regression outputs: [batch, 2] → mean, log_var
        pki_out = self.pki_head(h)
        pic50_out = self.pic50_head(h)
        bias_out = self.kinetic_bias_head(h)

        # Classification output
        mode_logits = self.mode_head(h)

        return {
            'pKi_mean': pki_out[:, 0],
            'pKi_log_var': pki_out[:, 1],
            'pIC50_mean': pic50_out[:, 0],
            'pIC50_log_var': pic50_out[:, 1],
            'kinetic_bias_mean': torch.sigmoid(bias_out[:, 0]),  # 0-1 range
            'kinetic_bias_log_var': bias_out[:, 1],
            'interaction_mode': mode_logits,  # 4-class logits
        }


class StereoGNNKinetic(StereoGNN):
    """
    Extended StereoGNN with kinetic parameter prediction.

    This model predicts both:
    1. Activity classification (substrate/blocker/inactive) - original functionality
    2. Mechanistic kinetic parameters (Ki, IC50, mode, bias) - new functionality

    Architecture preserves the full GNN backbone and adds parallel kinetic heads.
    The shared layer is expanded to accommodate additional representational capacity.

    Usage:
        model = StereoGNNKinetic()
        output = model(data, return_kinetics=True)

        # Activity predictions (original)
        dat_logits = output['DAT']  # [batch, 3]

        # Kinetic predictions (new)
        dat_pki = output['DAT_pKi_mean']  # [batch]
        dat_pki_var = output['DAT_pKi_log_var']  # [batch]
        dat_mode = output['DAT_interaction_mode']  # [batch, 4]
    """

    def __init__(self, config: ModelConfig = None, kinetic_hidden_dim: int = 128):
        # Initialize base class first
        super().__init__(config)

        self.kinetic_hidden_dim = kinetic_hidden_dim

        # Expand shared layer for additional capacity (256 → 384)
        self.shared_dim = 384
        self.shared_layer = nn.Sequential(
            nn.Linear(self.config.atom_hidden_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        # Update existing classification heads to accept expanded shared dim
        self.heads = nn.ModuleDict({
            'DAT': MultiTaskHead(
                self.shared_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            ),
            'NET': MultiTaskHead(
                self.shared_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            ),
            'SERT': MultiTaskHead(
                self.shared_dim,
                self.config.task_specific_dim,
                self.config.num_classes,
                self.config.dropout,
            ),
        })

        # Add kinetic heads for each transporter
        self.kinetic_heads = nn.ModuleDict({
            'DAT': KineticHead(self.shared_dim, kinetic_hidden_dim, self.config.dropout),
            'NET': KineticHead(self.shared_dim, kinetic_hidden_dim, self.config.dropout),
            'SERT': KineticHead(self.shared_dim, kinetic_hidden_dim, self.config.dropout),
        })

    def forward(
        self,
        data: Data,
        return_attention: bool = False,
        return_kinetics: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with optional kinetic outputs.

        Args:
            data: PyTorch Geometric Data object
            return_attention: Whether to return attention weights
            return_kinetics: Whether to return kinetic parameters

        Returns:
            Dict with classification logits and optionally kinetic parameters
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Encode nodes (stereo-aware)
        x = self.node_encoder(x)

        # Encode edges (stereo-aware)
        edge_attr = self.edge_encoder(edge_attr)

        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Readout
        graph_emb, node_attn = self.readout(x, batch)

        # Store for interpretability
        self.last_node_attention = node_attn

        # Shared representation (expanded)
        shared = self.shared_layer(graph_emb)

        # Original classification outputs
        output = {
            'DAT': self.heads['DAT'](shared),
            'NET': self.heads['NET'](shared),
            'SERT': self.heads['SERT'](shared),
        }

        # Kinetic outputs
        if return_kinetics:
            for task in ['DAT', 'NET', 'SERT']:
                kinetics = self.kinetic_heads[task](shared)
                for k, v in kinetics.items():
                    output[f'{task}_{k}'] = v

        if return_attention:
            output['node_attention'] = node_attn
            output['graph_embedding'] = graph_emb

        return output

    def predict_kinetics_with_uncertainty(
        self,
        data: Data,
        n_mc_samples: int = 30,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Full uncertainty quantification for kinetic predictions.

        Combines:
        1. Aleatoric uncertainty: Predicted data noise (from heteroscedastic output)
        2. Epistemic uncertainty: Model uncertainty (from MC Dropout)
        3. Total uncertainty: Combined (sqrt of sum of squares)

        Args:
            data: PyTorch Geometric Data object
            n_mc_samples: Number of MC Dropout samples

        Returns:
            Nested dict with per-task predictions and uncertainty estimates
        """
        self.train()  # Enable dropout

        # Storage for MC samples
        mc_predictions = {
            task: {
                'pKi': [],
                'pIC50': [],
                'kinetic_bias': [],
                'interaction_mode': [],
                'class_probs': [],
            } for task in ['DAT', 'NET', 'SERT']
        }

        with torch.no_grad():
            for _ in range(n_mc_samples):
                output = self.forward(data, return_kinetics=True)

                for task in ['DAT', 'NET', 'SERT']:
                    mc_predictions[task]['pKi'].append(output[f'{task}_pKi_mean'])
                    mc_predictions[task]['pIC50'].append(output[f'{task}_pIC50_mean'])
                    mc_predictions[task]['kinetic_bias'].append(output[f'{task}_kinetic_bias_mean'])
                    mc_predictions[task]['interaction_mode'].append(
                        F.softmax(output[f'{task}_interaction_mode'], dim=-1)
                    )
                    mc_predictions[task]['class_probs'].append(
                        F.softmax(output[task], dim=-1)
                    )

        self.eval()

        # Get final aleatoric estimates (single forward pass)
        with torch.no_grad():
            final_output = self.forward(data, return_kinetics=True)

        results = {}
        for task in ['DAT', 'NET', 'SERT']:
            results[task] = {}

            # Continuous kinetic parameters
            for param in ['pKi', 'pIC50', 'kinetic_bias']:
                stacked = torch.stack(mc_predictions[task][param], dim=0)

                mean = stacked.mean(dim=0)
                epistemic = stacked.std(dim=0)

                # Aleatoric from predicted log_var
                log_var_key = f'{task}_{param}_log_var'
                log_var = final_output[log_var_key]
                aleatoric = torch.sqrt(torch.exp(torch.clamp(log_var, -10, 10)))

                # Total uncertainty (sqrt of sum of variances)
                total = torch.sqrt(aleatoric**2 + epistemic**2)

                results[task][param] = {
                    'mean': mean,
                    'aleatoric': aleatoric,
                    'epistemic': epistemic,
                    'total': total,
                }

            # Mode classification uncertainty
            mode_stacked = torch.stack(mc_predictions[task]['interaction_mode'], dim=0)
            mean_probs = mode_stacked.mean(dim=0)
            results[task]['interaction_mode'] = {
                'mean_probs': mean_probs,
                'epistemic': mode_stacked.std(dim=0),
                'predicted_class': mean_probs.argmax(dim=-1),
                'entropy': -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1),
                'class_names': KineticHead.MODE_NAMES,
            }

            # Activity classification uncertainty
            class_stacked = torch.stack(mc_predictions[task]['class_probs'], dim=0)
            class_mean = class_stacked.mean(dim=0)
            results[task]['activity'] = {
                'mean_probs': class_mean,
                'epistemic': class_stacked.std(dim=0),
                'predicted_class': class_mean.argmax(dim=-1),
            }

        return results

    def get_kinetic_embedding(self, data: Data) -> Tensor:
        """
        Get the shared representation used for kinetic prediction.

        Useful for:
        - Visualization (UMAP/t-SNE)
        - Similarity analysis
        - Transfer learning
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)

        graph_emb, _ = self.readout(x, batch)
        shared = self.shared_layer(graph_emb)

        return shared

    @classmethod
    def from_pretrained(
        cls,
        base_model: StereoGNN,
        freeze_backbone: bool = False,
    ) -> 'StereoGNNKinetic':
        """
        Create kinetic model from pre-trained base StereoGNN.

        Args:
            base_model: Pre-trained StereoGNN model
            freeze_backbone: Whether to freeze GNN backbone weights

        Returns:
            StereoGNNKinetic with transferred weights
        """
        kinetic_model = cls(config=base_model.config)

        # Transfer encoder weights
        kinetic_model.node_encoder.load_state_dict(base_model.node_encoder.state_dict())
        kinetic_model.edge_encoder.load_state_dict(base_model.edge_encoder.state_dict())

        # Transfer GNN layer weights
        for i, layer in enumerate(base_model.gnn_layers):
            kinetic_model.gnn_layers[i].load_state_dict(layer.state_dict())

        # Transfer readout weights
        kinetic_model.readout.load_state_dict(base_model.readout.state_dict())

        if freeze_backbone:
            for param in kinetic_model.node_encoder.parameters():
                param.requires_grad = False
            for param in kinetic_model.edge_encoder.parameters():
                param.requires_grad = False
            for layer in kinetic_model.gnn_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            for param in kinetic_model.readout.parameters():
                param.requires_grad = False

        return kinetic_model


if __name__ == "__main__":
    print("=" * 60)
    print("StereoGNN Model Test")
    print("=" * 60)

    # Create model
    model = StereoGNN()
    print(f"Total parameters: {count_parameters(model):,}")

    # Create dummy data
    from featurizer import MoleculeGraphFeaturizer

    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    test_smiles = [
        "C[C@H](N)Cc1ccccc1",  # d-Amphetamine
        "C[C@@H](N)Cc1ccccc1",  # l-Amphetamine
    ]

    data_list = [featurizer.featurize(smi) for smi in test_smiles]

    # Batch
    batch = Batch.from_data_list(data_list)
    print(f"\nBatch: {batch}")
    print(f"Node features shape: {batch.x.shape}")
    print(f"Edge features shape: {batch.edge_attr.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch, return_attention=True)

    print("\nOutput shapes:")
    for key, val in output.items():
        if isinstance(val, Tensor):
            print(f"  {key}: {val.shape}")

    # Test uncertainty estimation
    print("\nUncertainty estimation:")
    uncertainty = model.predict_with_uncertainty(batch, n_samples=10)
    for task, result in uncertainty.items():
        print(f"  {task}: mean={result['mean'].shape}, std={result['std'].shape}")

    # Ablation model
    print("\n" + "=" * 60)
    print("Ablation Model (no stereo features)")
    ablation_model = StereoGNNForAblation()
    print(f"Ablation parameters: {count_parameters(ablation_model):,}")
