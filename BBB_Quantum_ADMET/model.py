"""
Quantum-Aware GNN Model for Multi-Task ADMET Prediction

Architecture:
- QuantumAwareEncoder: 5-layer GATv2 + Transformer with 34 input features
- MultiTaskHead: Shared-bottom architecture for 6 ADMET endpoints
- Regression outputs (not classification)

This is INDEPENDENT from BBB_System.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    TransformerConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    BatchNorm
)
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple

from config import MODEL_CONFIG, TRAINING_CONFIG


class QuantumAwareEncoder(nn.Module):
    """
    Graph encoder for 34-dimensional quantum-enhanced features.

    Architecture:
    - Input embedding (34 -> hidden_dim)
    - 5x GATv2 layers with residual connections
    - 1x TransformerConv layer
    - Mean + Max pooling for graph-level embedding
    """

    def __init__(
        self,
        node_features: int = 34,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Feature-specific embeddings for quantum features
        self.quantum_embed = nn.Sequential(
            nn.Linear(13, hidden_dim // 4),  # Quantum features 16-28
            nn.GELU()
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()

        for i in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True,
                    share_weights=False
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            self.gat_dropouts.append(nn.Dropout(dropout))

        # Transformer layer for global attention
        self.transformer = TransformerConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, 34]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embeddings [batch_size, hidden_dim * 2]
        """
        # Initial embedding
        h = self.input_embed(x)

        # GAT layers with residual connections
        for gat, norm, drop in zip(self.gat_layers, self.gat_norms, self.gat_dropouts):
            h_new = gat(h, edge_index)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = drop(h_new)
            h = h + h_new  # Residual

        # Transformer layer
        h_trans = self.transformer(h, edge_index)
        h_trans = self.transformer_norm(h_trans)
        h = h + h_trans  # Residual

        # Graph-level pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)

        # Concatenate pooling strategies
        graph_embed = torch.cat([h_mean, h_max], dim=-1)

        # Final projection
        graph_embed = self.output_proj(graph_embed)

        return graph_embed


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for ADMET endpoints.

    Architecture:
    - Shared bottom layers
    - Task-specific towers
    - Regression outputs
    """

    def __init__(
        self,
        input_dim: int = 512,  # hidden_dim * 2 from encoder
        hidden_dim: int = 256,
        num_tasks: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_tasks = num_tasks

        # Shared bottom
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )

        # Task-specific towers
        self.task_towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)  # Regression output
            )
            self.task_towers.append(tower)

        # Task names for reference
        self.task_names = ['logbb', 'logs', 'logp', 'cyp3a4', 'herg', 'ld50']

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Graph embeddings [batch_size, input_dim]

        Returns:
            Dict mapping task names to predictions [batch_size, 1]
        """
        # Shared representation
        shared_repr = self.shared(x)

        # Task-specific predictions
        outputs = {}
        for i, (name, tower) in enumerate(zip(self.task_names, self.task_towers)):
            outputs[name] = tower(shared_repr)

        return outputs


class QuantumADMETModel(nn.Module):
    """
    Complete model: Encoder + Multi-Task Head

    For training and inference on ADMET endpoints.
    """

    def __init__(
        self,
        node_features: int = 34,
        hidden_dim: int = 256,
        num_gat_layers: int = 5,
        num_heads: int = 8,
        num_tasks: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = QuantumAwareEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.head = MultiTaskHead(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout + 0.1
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, 34]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Dict of task predictions
        """
        graph_embed = self.encoder(x, edge_index, batch)
        predictions = self.head(graph_embed)
        return predictions

    def predict_single(self, data: Data) -> Dict[str, float]:
        """
        Predict ADMET values for a single molecule.

        Args:
            data: PyG Data object

        Returns:
            Dict of predicted values
        """
        self.eval()
        with torch.no_grad():
            batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            outputs = self.forward(data.x, data.edge_index, batch)
            return {k: v.item() for k, v in outputs.items()}


class PretrainingHead(nn.Module):
    """
    Self-supervised pretraining head for quantum features.

    Tasks:
    - Predict molecular weight
    - Predict atom count
    - Predict LogP
    - Predict has stereocenters
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()

        self.mol_weight_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.atom_count_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.logp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.stereo_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass returning all predictions."""
        return (
            self.mol_weight_head(x),
            self.atom_count_head(x),
            self.logp_head(x),
            self.stereo_head(x)
        )


class QuantumPretrainingModel(nn.Module):
    """
    Model for self-supervised pretraining on ZINC.
    """

    def __init__(
        self,
        node_features: int = 34,
        hidden_dim: int = 256,
        num_gat_layers: int = 5,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.encoder = QuantumAwareEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.pretrain_head = PretrainingHead(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim // 2
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for pretraining."""
        graph_embed = self.encoder(x, edge_index, batch)
        return self.pretrain_head(graph_embed)

    def get_encoder(self) -> QuantumAwareEncoder:
        """Get the encoder for fine-tuning."""
        return self.encoder


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test
if __name__ == "__main__":
    print("Testing Quantum ADMET Model...")
    print("=" * 60)

    # Create model
    model = QuantumADMETModel(
        node_features=34,
        hidden_dim=256,
        num_gat_layers=5,
        num_heads=8,
        num_tasks=6
    )

    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Encoder parameters: {count_parameters(model.encoder):,}")
    print(f"Head parameters: {count_parameters(model.head):,}")

    # Test forward pass
    batch_size = 4
    num_nodes = 20
    num_edges = 40

    x = torch.randn(num_nodes * batch_size, 34)
    edge_index = torch.randint(0, num_nodes, (2, num_edges * batch_size))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)

    outputs = model(x, edge_index, batch)

    print(f"\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tensor.shape}")

    print("\n" + "=" * 60)
    print("Model test complete!")
