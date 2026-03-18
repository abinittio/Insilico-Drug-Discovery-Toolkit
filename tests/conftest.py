"""Shared fixtures for the Insilico Drug Discovery Toolkit test suite."""

import sys
from pathlib import Path

import pytest

# Add subproject directories to sys.path so their modules are importable
ROOT = Path(__file__).resolve().parent.parent
for subdir in ("StereoGNN_Transporter", "BBB_System", "BBB_Quantum_ADMET"):
    path = ROOT / subdir
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ---------------------------------------------------------------------------
# Sample SMILES used across tests
# ---------------------------------------------------------------------------
SAMPLE_SMILES = [
    "C[C@H](N)Cc1ccccc1",   # d-Amphetamine
    "C[C@@H](N)Cc1ccccc1",  # l-Amphetamine
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "c1ccccc1",               # Benzene
    "CCO",                    # Ethanol
]


@pytest.fixture
def sample_smiles() -> list[str]:
    """Return a list of well-known, RDKit-parseable SMILES strings."""
    return list(SAMPLE_SMILES)
