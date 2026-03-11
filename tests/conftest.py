"""Shared fixtures for StereoGNN test suite."""

import pytest


SAMPLE_SMILES = {
    "amphetamine": "C[C@H](N)Cc1ccccc1",
    "cocaine": "COC(=O)C1CC2CCC(C1)N2C",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "invalid": "NOT_A_SMILES",
    "empty": "",
}


@pytest.fixture
def sample_smiles():
    return SAMPLE_SMILES
