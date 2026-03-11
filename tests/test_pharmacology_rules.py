"""Smoke tests for the pharmacology rules engine."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pharmacology_rules import PharmacologyRules


@pytest.fixture
def rules():
    return PharmacologyRules()


class TestPharmacologyRules:
    """Tests for PharmacologyRules."""

    def test_rules_initialise(self, rules):
        assert rules is not None

    def test_has_patterns(self, rules):
        assert hasattr(rules, "PATTERNS")
        assert len(rules.PATTERNS) > 0

    def test_correct_predictions_returns_dict(self, rules, sample_smiles):
        mock_preds = {
            "DAT": {"class": "inactive", "probs": [0.1, 0.1, 0.8]},
            "NET": {"class": "inactive", "probs": [0.1, 0.1, 0.8]},
            "SERT": {"class": "inactive", "probs": [0.1, 0.1, 0.8]},
        }
        result = rules.correct_predictions(sample_smiles["amphetamine"], mock_preds)
        assert isinstance(result, dict)

    def test_caffeine_no_crash(self, rules, sample_smiles):
        mock_preds = {
            "DAT": {"class": "inactive", "probs": [0.1, 0.1, 0.8]},
            "NET": {"class": "inactive", "probs": [0.1, 0.1, 0.8]},
            "SERT": {"class": "inactive", "probs": [0.1, 0.1, 0.8]},
        }
        result = rules.correct_predictions(sample_smiles["caffeine"], mock_preds)
        assert result is not None
