"""Smoke tests for the StereoGNN Transporter model."""

import torch
from torch_geometric.data import Data, Batch

from model import StereoGNN, StereoGNNKinetic, StereoGNNForAblation, count_parameters
from featurizer import MoleculeGraphFeaturizer, StereoFeaturizer, get_feature_dimensions
from losses import FocalLoss, MultiTaskLoss, KineticMultiTaskLoss
from config import CONFIG


# ---------------------------------------------------------------------------
# Featurizer tests
# ---------------------------------------------------------------------------

class TestFeaturizer:
    """Tests for molecular featurization."""

    def test_featurize_valid_smiles(self, sample_smiles: list[str]) -> None:
        featurizer = MoleculeGraphFeaturizer(use_3d=False)
        for smi in sample_smiles:
            data = featurizer.featurize(smi)
            assert data is not None, f"Featurization failed for {smi}"
            assert data.x.dim() == 2
            assert data.edge_index.dim() == 2
            assert data.edge_index.shape[0] == 2

    def test_featurize_invalid_smiles(self) -> None:
        featurizer = MoleculeGraphFeaturizer(use_3d=False)
        result = featurizer.featurize("NOT_A_SMILES")
        assert result is None

    def test_feature_dimensions(self) -> None:
        dims = get_feature_dimensions()
        assert dims["node_features"] > 0
        assert dims["edge_features"] > 0

    def test_stereo_features_differ_for_enantiomers(self) -> None:
        featurizer = MoleculeGraphFeaturizer(use_3d=False)
        d_amp = featurizer.featurize("C[C@H](N)Cc1ccccc1")
        l_amp = featurizer.featurize("C[C@@H](N)Cc1ccccc1")
        assert d_amp is not None and l_amp is not None
        # Node features should differ due to R/S encoding
        assert not torch.equal(d_amp.x, l_amp.x)

    def test_batch_featurize(self, sample_smiles: list[str]) -> None:
        featurizer = MoleculeGraphFeaturizer(use_3d=False)
        data_list = featurizer.featurize_batch(sample_smiles)
        assert len(data_list) == len(sample_smiles)


# ---------------------------------------------------------------------------
# Model instantiation tests
# ---------------------------------------------------------------------------

class TestModelInstantiation:
    """Tests that models instantiate and forward-pass without errors."""

    def _make_batch(self) -> Batch:
        """Create a small batch from real SMILES."""
        featurizer = MoleculeGraphFeaturizer(use_3d=False)
        data_list = [
            featurizer.featurize("C[C@H](N)Cc1ccccc1"),
            featurizer.featurize("c1ccccc1"),
        ]
        return Batch.from_data_list(data_list)

    def test_stereognn_instantiates(self) -> None:
        model = StereoGNN()
        assert count_parameters(model) > 0

    def test_stereognn_forward(self) -> None:
        model = StereoGNN()
        model.eval()
        batch = self._make_batch()
        with torch.no_grad():
            output = model(batch)
        for target in ("DAT", "NET", "SERT"):
            assert target in output
            assert output[target].shape == (2, 3)  # batch=2, classes=3

    def test_stereognn_kinetic_forward(self) -> None:
        model = StereoGNNKinetic()
        model.eval()
        batch = self._make_batch()
        with torch.no_grad():
            output = model(batch, return_kinetics=True)
        for target in ("DAT", "NET", "SERT"):
            assert output[target].shape == (2, 3)
            assert f"{target}_pKi_mean" in output
            assert output[f"{target}_pKi_mean"].shape == (2,)

    def test_ablation_model_forward(self) -> None:
        model = StereoGNNForAblation()
        model.eval()
        batch = self._make_batch()
        with torch.no_grad():
            output = model(batch)
        for target in ("DAT", "NET", "SERT"):
            assert output[target].shape == (2, 3)

    def test_uncertainty_estimation(self) -> None:
        model = StereoGNN()
        batch = self._make_batch()
        results = model.predict_with_uncertainty(batch, n_samples=5)
        for target in ("DAT", "NET", "SERT"):
            assert "mean" in results[target]
            assert "std" in results[target]
            assert results[target]["mean"].shape == (2, 3)

    def test_get_embedding(self) -> None:
        model = StereoGNN()
        model.eval()
        batch = self._make_batch()
        with torch.no_grad():
            emb = model.get_embedding(batch)
        assert emb.dim() == 2
        assert emb.shape[0] == 2  # batch size


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLosses:
    """Tests for custom loss functions."""

    def test_focal_loss(self) -> None:
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_focal_loss_ignore_index(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, ignore_index=-1)
        logits = torch.randn(4, 3)
        labels = torch.tensor([0, 1, -1, 2])
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0

    def test_multi_task_loss(self) -> None:
        mt_loss = MultiTaskLoss(loss_fn="focal", learn_weights=True)
        preds = {t: torch.randn(4, 3) for t in ("DAT", "NET", "SERT")}
        targets = {t: torch.randint(0, 3, (4,)) for t in ("DAT", "NET", "SERT")}
        losses = mt_loss(preds, targets)
        assert "total" in losses
        assert losses["total"].item() >= 0
