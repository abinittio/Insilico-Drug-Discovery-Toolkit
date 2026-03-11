"""Smoke tests for configuration loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG, Config


class TestConfig:
    """Tests for configuration dataclasses."""

    def test_config_loads(self):
        assert CONFIG is not None

    def test_config_is_correct_type(self):
        assert isinstance(CONFIG, Config)

    def test_paths_are_relative(self):
        """Config paths should not be hardcoded to a specific user directory."""
        root = CONFIG.data.project_root
        assert root.is_absolute(), "Project root should resolve to absolute path"
        assert root.exists(), f"Project root does not exist: {root}"

    def test_data_dir_exists(self):
        assert CONFIG.data.data_dir.exists()

    def test_models_dir_exists(self):
        assert CONFIG.data.models_dir.exists()

    def test_model_hyperparameters(self):
        assert CONFIG.model.num_gnn_layers > 0
        assert CONFIG.model.num_attention_heads > 0
        assert 0 < CONFIG.model.dropout < 1

    def test_training_hyperparameters(self):
        assert CONFIG.training.learning_rate > 0
        assert CONFIG.training.max_epochs > 0
        assert CONFIG.training.patience > 0

    def test_chembl_targets_defined(self):
        targets = CONFIG.data.chembl_targets
        assert "DAT" in targets
        assert "NET" in targets
        assert "SERT" in targets
