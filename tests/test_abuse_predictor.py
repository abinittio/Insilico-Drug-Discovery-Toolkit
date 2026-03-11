"""Smoke tests for the abuse liability prediction engine."""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abuse_predictor import AbusePredictor


@pytest.fixture
def predictor():
    return AbusePredictor()


# Default MAT probabilities for testing (3-class: substrate, blocker, inactive)
DUMMY_PROBS = np.array([0.33, 0.33, 0.34])


class TestAbusePredictor:
    """Tests for AbusePredictor."""

    def test_predictor_initialises(self, predictor):
        assert predictor is not None

    def test_amphetamine_is_high_abuse(self, predictor, sample_smiles):
        score, category = predictor.predict(
            sample_smiles["amphetamine"],
            dat_class="substrate", net_class="substrate", sert_class="substrate",
            dat_probs=DUMMY_PROBS, net_probs=DUMMY_PROBS, sert_probs=DUMMY_PROBS,
        )
        assert category == "HIGH", f"Amphetamine should be HIGH abuse, got {category}"

    def test_caffeine_is_low_abuse(self, predictor, sample_smiles):
        score, category = predictor.predict(
            sample_smiles["caffeine"],
            dat_class="inactive", net_class="inactive", sert_class="inactive",
            dat_probs=DUMMY_PROBS, net_probs=DUMMY_PROBS, sert_probs=DUMMY_PROBS,
        )
        assert category == "LOW", f"Caffeine should be LOW abuse, got {category}"

    def test_returns_tuple(self, predictor, sample_smiles):
        result = predictor.predict(
            sample_smiles["amphetamine"],
            dat_class="substrate", net_class="substrate", sert_class="substrate",
            dat_probs=DUMMY_PROBS, net_probs=DUMMY_PROBS, sert_probs=DUMMY_PROBS,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_in_valid_range(self, predictor, sample_smiles):
        score, category = predictor.predict(
            sample_smiles["amphetamine"],
            dat_class="substrate", net_class="substrate", sert_class="substrate",
            dat_probs=DUMMY_PROBS, net_probs=DUMMY_PROBS, sert_probs=DUMMY_PROBS,
        )
        assert 0 <= score <= 100

    def test_category_is_valid(self, predictor, sample_smiles):
        score, category = predictor.predict(
            sample_smiles["aspirin"],
            dat_class="inactive", net_class="inactive", sert_class="inactive",
            dat_probs=DUMMY_PROBS, net_probs=DUMMY_PROBS, sert_probs=DUMMY_PROBS,
        )
        assert category in {"HIGH", "MODERATE", "LOW"}
