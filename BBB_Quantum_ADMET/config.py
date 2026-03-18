"""
Configuration for BBB Quantum ADMET Predictor
Multi-task regression with 34-dimensional quantum features
"""

from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Feature dimensions
    node_features: int = 34          # 15 atomic + 13 quantum + 6 stereo
    hidden_dim: int = 256            # Larger for quantum features
    num_gat_layers: int = 5          # Deeper network
    num_heads: int = 8               # More attention heads
    dropout: float = 0.2

    # Pooling
    pooling: str = "mean_max"        # Concatenate mean and max pooling

    # Output
    num_tasks: int = 6               # Multi-task ADMET endpoints


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Pretraining
    pretrain_epochs: int = 30
    pretrain_batch_size: int = 128
    pretrain_lr: float = 0.001

    # Fine-tuning
    finetune_epochs: int = 100
    finetune_batch_size: int = 64
    finetune_lr: float = 0.0005

    # Optimization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    scheduler: str = "cosine"        # cosine, plateau, or step
    warmup_epochs: int = 5

    # Multi-task learning
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'logbb': 1.0,      # Primary task
        'logs': 0.8,       # Solubility
        'logp': 0.5,       # Lipophilicity (easier)
        'cyp3a4': 0.8,     # CYP inhibition
        'herg': 1.0,       # Cardiac toxicity (important)
        'ld50': 0.7,       # Acute toxicity
    })

    # Validation
    n_folds: int = 5
    early_stopping_patience: int = 15

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DataConfig:
    """Data configuration."""
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    checkpoint_dir: str = "models/checkpoints"
    results_dir: str = "results"

    # Preprocessing
    max_atoms: int = 150             # Max atoms per molecule
    max_conformers: int = 1          # ETKDG conformers to generate
    etkdg_random_seed: int = 42

    # ADMET endpoints
    endpoints: List[str] = field(default_factory=lambda: [
        'logbb',    # Blood-brain barrier (log scale)
        'logs',     # Aqueous solubility (log mol/L)
        'logp',     # Octanol-water partition
        'cyp3a4',   # CYP3A4 pKi
        'herg',     # hERG pIC50
        'ld50',     # LD50 (log mg/kg)
    ])

    # Normalization targets (approximate ranges for standardization)
    endpoint_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        'logbb': (-3.0, 2.0),     # Log BB ratio
        'logs': (-10.0, 2.0),     # Log solubility
        'logp': (-3.0, 8.0),      # LogP
        'cyp3a4': (3.0, 10.0),    # pKi
        'herg': (3.0, 9.0),       # pIC50
        'ld50': (1.0, 5.0),       # Log LD50
    })


@dataclass
class QuantumConfig:
    """Quantum descriptor configuration."""
    # ETKDG settings
    use_etkdg: bool = True
    num_conformers: int = 1
    max_iterations: int = 200
    prune_rms_threshold: float = 0.5
    random_seed: int = 42

    # Quantum approximation method
    method: str = "rdkit"            # rdkit or mordred

    # Feature scaling
    scale_features: bool = True

    # Caching
    cache_conformers: bool = True
    cache_dir: str = "data/conformer_cache"


# Default configurations
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
QUANTUM_CONFIG = QuantumConfig()


# Electronegativity values (Pauling scale)
ELECTRONEGATIVITY = {
    1: 2.20,   # H
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}

# Atomic polarizabilities (Angstrom^3)
POLARIZABILITY = {
    1: 0.667,   # H
    6: 1.76,    # C
    7: 1.10,    # N
    8: 0.802,   # O
    9: 0.557,   # F
    15: 3.63,   # P
    16: 2.90,   # S
    17: 2.18,   # Cl
    35: 3.05,   # Br
    53: 5.35,   # I
}

# Ionization energies (eV) - for HOMO approximation
IONIZATION_ENERGY = {
    1: 13.60,   # H
    6: 11.26,   # C
    7: 14.53,   # N
    8: 13.62,   # O
    9: 17.42,   # F
    15: 10.49,  # P
    16: 10.36,  # S
    17: 12.97,  # Cl
    35: 11.81,  # Br
    53: 10.45,  # I
}

# Electron affinities (eV) - for LUMO approximation
ELECTRON_AFFINITY = {
    1: 0.75,    # H
    6: 1.26,    # C
    7: -0.07,   # N (negative = doesn't bind)
    8: 1.46,    # O
    9: 3.40,    # F
    15: 0.75,   # P
    16: 2.08,   # S
    17: 3.61,   # Cl
    35: 3.36,   # Br
    53: 3.06,   # I
}
