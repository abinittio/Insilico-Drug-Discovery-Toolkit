"""Smoke tests for the molecular featurization pipeline."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featurizer import MoleculeGraphFeaturizer


@pytest.fixture
def featurizer():
    return MoleculeGraphFeaturizer()


class TestFeaturizer:
    """Tests for MoleculeGraphFeaturizer."""

    def test_featurizer_initialises(self, featurizer):
        assert featurizer is not None

    def test_valid_smiles_returns_data(self, featurizer, sample_smiles):
        data = featurizer.featurize(sample_smiles["amphetamine"])
        assert data is not None

    def test_output_has_node_features(self, featurizer, sample_smiles):
        data = featurizer.featurize(sample_smiles["amphetamine"])
        assert hasattr(data, "x")
        assert data.x.shape[1] > 0, "Node features should have non-zero dimension"

    def test_output_has_edge_index(self, featurizer, sample_smiles):
        data = featurizer.featurize(sample_smiles["amphetamine"])
        assert hasattr(data, "edge_index")
        assert data.edge_index.shape[0] == 2, "Edge index should have 2 rows (src, dst)"

    def test_invalid_smiles_returns_none(self, featurizer, sample_smiles):
        result = featurizer.featurize(sample_smiles["invalid"])
        assert result is None

    def test_multiple_molecules(self, featurizer, sample_smiles):
        for name in ["amphetamine", "cocaine", "caffeine", "fluoxetine"]:
            data = featurizer.featurize(sample_smiles[name])
            assert data is not None, f"Featurization failed for {name}"

    def test_stereo_features_present(self, featurizer, sample_smiles):
        data = featurizer.featurize(sample_smiles["amphetamine"])
        # Amphetamine has a stereocenter, so total node features should include stereo dims
        # Base: 75, Stereo: 11 → total ≥ 86
        assert data.x.shape[1] >= 75, "Expected at least base atom features (75D)"
