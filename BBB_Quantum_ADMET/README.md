# BBB Quantum ADMET Predictor

## Multi-Task Graph Neural Network with Quantum Descriptors

**Project Type:** Research / Development
**Started:** December 2025
**Status:** In Development

---

## Overview

This project extends molecular property prediction using:

1. **34-dimensional quantum-enhanced node features** (15 atomic + 13 quantum + 6 stereo)
2. **RDKit ETKDG 3D conformer generation** for geometry-dependent features
3. **Multi-task regression** for ADMET endpoints
4. **Separate architecture** from BBB_System baseline

---

## Key Differences from BBB_System (v1.0)

| Aspect | BBB_System | BBB_Quantum_ADMET |
|--------|------------|-------------------|
| Features | 21 (atomic + stereo) | 34 (atomic + quantum + stereo) |
| 3D Method | None | ETKDG conformers |
| Task | Binary classification | Multi-task regression |
| Endpoints | BBB only | BBB + Solubility + CYP + Toxicity |
| Output | Probability | Continuous values |

---

## Feature Set (34 dimensions)

### Atomic Features (1-15)
- Atomic number, degree, formal charge
- Hybridization, aromaticity, ring membership
- Implicit H count, total valence, atomic mass
- Electronegativity, polarity, H-bond donor/acceptor
- Partial charge approximation, lipophilic contribution

### Quantum Descriptors (16-28) - ETKDG-derived
- HOMO/LUMO energy approximations
- Orbital gap
- Mulliken charge estimates
- Polarizability
- Electrophilicity/Nucleophilicity indices
- Fukui functions (f+, f-, f0)
- Softness parameters
- Hardness parameters
- Electrostatic potential features

### Stereochemistry (29-34)
- Is chiral center
- R configuration
- S configuration
- Part of E/Z bond
- E configuration
- Z configuration

---

## ADMET Endpoints (Multi-Task Regression)

1. **LogBB** - Blood-brain barrier partition coefficient
2. **LogS** - Aqueous solubility
3. **LogP** - Lipophilicity
4. **CYP3A4 pKi** - Cytochrome P450 inhibition
5. **hERG pIC50** - Cardiac toxicity risk
6. **LD50** - Acute toxicity

---

## Directory Structure

```
BBB_Quantum_ADMET/
├── README.md
├── config.py                 # Hyperparameters and settings
├── quantum_features.py       # 34-dim feature extraction with ETKDG
├── model.py                  # QuantumAwareEncoder + MultiTaskHead
├── datasets.py               # ADMET data loading and processing
├── train.py                  # Multi-task training loop
├── evaluate.py               # Per-task metrics
├── predict.py                # Inference script
├── data/
│   ├── admet_combined.csv    # Multi-endpoint training data
│   └── preprocessed/         # Cached graphs
├── models/
│   └── checkpoints/          # Training checkpoints
└── results/
    └── metrics/              # Evaluation results
```

---

## Requirements

```
torch>=2.0
torch_geometric>=2.3
rdkit>=2023.03
pandas
numpy
scikit-learn
```

---

## Notes

- This project is INDEPENDENT from BBB_System
- No shared model weights or training data
- Different architecture optimized for regression
- Focus on quantum-mechanical descriptors via ETKDG approximations
