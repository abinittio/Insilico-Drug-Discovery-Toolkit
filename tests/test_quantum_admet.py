"""Smoke tests for the BBB Quantum ADMET model."""

import torch
from torch_geometric.data import Data, Batch

from model import QuantumAwareEncoder
from config import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Model instantiation tests
# ---------------------------------------------------------------------------

def _make_dummy_batch(
    num_graphs: int = 3,
    num_node_features: int = 34,
) -> Batch:
    """Create a dummy batch with random node features."""
    data_list = []
    for _ in range(num_graphs):
        n_atoms = torch.randint(5, 20, (1,)).item()
        n_edges = torch.randint(4, 30, (1,)).item()
        data = Data(
            x=torch.randn(n_atoms, num_node_features),
            edge_index=torch.randint(0, n_atoms, (2, n_edges)),
            batch=torch.zeros(n_atoms, dtype=torch.long),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


class TestQuantumAwareEncoder:
    """Tests for the QuantumAwareEncoder."""

    def test_instantiates(self) -> None:
        model = QuantumAwareEncoder(
            node_features=MODEL_CONFIG.node_features,
            hidden_dim=MODEL_CONFIG.hidden_dim,
            num_layers=MODEL_CONFIG.num_gat_layers,
            num_heads=MODEL_CONFIG.num_heads,
            dropout=MODEL_CONFIG.dropout,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_forward_shape(self) -> None:
        model = QuantumAwareEncoder(
            node_features=34,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )
        model.eval()
        batch = _make_dummy_batch(num_graphs=2, num_node_features=34)
        with torch.no_grad():
            emb = model(batch.x, batch.edge_index, batch.batch)
        # Output should be [num_graphs, hidden_dim * 2] due to mean+max pooling
        assert emb.dim() == 2
        assert emb.shape[0] == 2

    def test_config_defaults(self) -> None:
        assert MODEL_CONFIG.node_features == 34
        assert MODEL_CONFIG.num_tasks == 6
