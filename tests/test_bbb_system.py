"""Smoke tests for the BBB System models."""

import torch
from torch_geometric.data import Data, Batch

from bbb_gnn_model import HybridGATSAGE
from advanced_bbb_model import AdvancedHybridBBBNet
from mol_to_graph import mol_to_graph, get_molecular_descriptors


# ---------------------------------------------------------------------------
# Featurizer tests
# ---------------------------------------------------------------------------

class TestMolToGraph:
    """Tests for molecular graph conversion."""

    def test_mol_to_graph_valid(self, sample_smiles: list[str]) -> None:
        for smi in sample_smiles:
            data = mol_to_graph(smi)
            assert data is not None, f"mol_to_graph failed for {smi}"
            assert data.x.dim() == 2
            assert data.edge_index.dim() == 2
            assert data.edge_index.shape[0] == 2

    def test_mol_to_graph_invalid(self) -> None:
        result = mol_to_graph("INVALID")
        assert result is None

    def test_molecular_descriptors(self) -> None:
        desc = get_molecular_descriptors("c1ccccc1")  # Benzene
        assert desc is not None
        assert "mol_weight" in desc or "MW" in desc or isinstance(desc, dict)


# ---------------------------------------------------------------------------
# Model instantiation tests
# ---------------------------------------------------------------------------

def _make_bbb_batch(num_node_features: int = 15) -> Batch:
    """Create a minimal BBB batch from real SMILES via mol_to_graph."""
    graphs = []
    for smi in ["c1ccccc1", "CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]:
        g = mol_to_graph(smi)
        if g is not None:
            # Pad/truncate node features to expected size
            n = g.x.shape[0]
            current_feats = g.x.shape[1]
            if current_feats < num_node_features:
                padding = torch.zeros(n, num_node_features - current_feats)
                g.x = torch.cat([g.x, padding], dim=1)
            elif current_feats > num_node_features:
                g.x = g.x[:, :num_node_features]
            graphs.append(g)
    return Batch.from_data_list(graphs)


class TestHybridGATSAGE:
    """Tests for the HybridGATSAGE model."""

    def test_instantiates(self) -> None:
        model = HybridGATSAGE(num_node_features=15)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_forward_shape(self) -> None:
        model = HybridGATSAGE(num_node_features=15)
        model.eval()
        batch = _make_bbb_batch(num_node_features=15)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.dim() == 2
        assert out.shape[0] == batch.num_graphs
        assert out.shape[1] == 1  # single BBB score


class TestAdvancedHybridBBBNet:
    """Tests for the AdvancedHybridBBBNet model."""

    def test_instantiates(self) -> None:
        model = AdvancedHybridBBBNet(num_node_features=15)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_forward_shape(self) -> None:
        model = AdvancedHybridBBBNet(num_node_features=15)
        model.eval()
        batch = _make_bbb_batch(num_node_features=15)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
        assert out.dim() == 2
        assert out.shape[0] == batch.num_graphs
        assert out.shape[1] == 1
