"""
StereoGNN Transporter Substrate Predictor - Configuration
=========================================================

Central configuration for the monoamine transporter substrate prediction system.
Targets: DAT (Dopamine), NET (Norepinephrine), SERT (Serotonin)

Success Criteria (MUST ACHIEVE):
- Overall ROC-AUC: >= 0.85
- Monoamine-specific ROC-AUC: >= 0.95
- PR-AUC: >= 0.65
- Stereo selectivity: >= 80% correct on known enantiomer pairs
- Ablation drop (no stereo): >= 5%
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch

# Note: Tuple is imported above for type hints in KineticConfig


@dataclass
class DataConfig:
    """Data curation and preprocessing configuration."""

    # Project paths
    project_root: Path = Path("C:/Users/nakhi/StereoGNN_Transporter")
    data_dir: Path = field(default_factory=lambda: Path("C:/Users/nakhi/StereoGNN_Transporter/data"))
    models_dir: Path = field(default_factory=lambda: Path("C:/Users/nakhi/StereoGNN_Transporter/models"))
    results_dir: Path = field(default_factory=lambda: Path("C:/Users/nakhi/StereoGNN_Transporter/results"))

    # ChEMBL target IDs for monoamine transporters
    # These are the UniProt accessions used in ChEMBL
    chembl_targets: Dict[str, str] = field(default_factory=lambda: {
        "DAT": "CHEMBL238",   # SLC6A3 - Dopamine transporter
        "NET": "CHEMBL222",   # SLC6A2 - Norepinephrine transporter
        "SERT": "CHEMBL228",  # SLC6A4 - Serotonin transporter
    })

    # Assay type filters - CRITICAL for substrate vs blocker distinction
    # We want UPTAKE assays, not just binding
    substrate_assay_keywords: List[str] = field(default_factory=lambda: [
        "uptake",
        "transport",
        "translocation",
        "efflux",
        "release",  # For releaser-type substrates
        "Km",
        "Vmax",
    ])

    # Assay types to EXCLUDE (these measure binding, not transport)
    blocker_assay_keywords: List[str] = field(default_factory=lambda: [
        "binding",
        "Ki",
        "IC50",
        "displacement",
        "radioligand",
        "competition",
    ])

    # Label definition thresholds
    # For uptake assays: % of control uptake remaining
    # Substrate: causes efflux OR is transported (reduces uptake by being transported)
    # Blocker: inhibits uptake without being transported
    substrate_threshold: float = 50.0  # < 50% uptake = active compound
    potent_threshold: float = 10.0     # < 10 uM = potent

    # Minimum data requirements
    min_substrates_per_target: int = 50
    min_total_compounds: int = 500

    # Data quality filters
    max_molecular_weight: float = 600.0  # CNS drug-like
    min_molecular_weight: float = 100.0
    max_heavy_atoms: int = 50
    min_heavy_atoms: int = 5

    # Scaffold split configuration
    test_fraction: float = 0.15
    val_fraction: float = 0.15
    scaffold_split_seed: int = 42


@dataclass
class StereoConfig:
    """Stereochemistry encoding configuration."""

    # Chirality encoding dimensions
    chiral_tag_dim: int = 8      # CW, CCW, unspecified, etc.
    stereo_bond_dim: int = 8     # E/Z, cis/trans

    # 3D conformation handling
    use_3d_coords: bool = True
    num_conformers: int = 5      # Generate multiple conformers for ensemble

    # Tetrahedral stereocenters
    encode_tetrahedral: bool = True
    encode_rs_config: bool = True  # R/S absolute configuration

    # Double bond geometry
    encode_ez_geometry: bool = True

    # Axial chirality (biaryl systems)
    encode_axial: bool = True

    # Known stereoselective pairs for validation
    # Format: (SMILES_1, SMILES_2, target, expected_ratio)
    # expected_ratio > 1 means first is more potent
    stereoselective_pairs: List[Tuple[str, str, str, float]] = field(default_factory=lambda: [
        # d-amphetamine vs l-amphetamine for DAT (d >> l, ~10x)
        ("C[C@H](N)Cc1ccccc1", "C[C@@H](N)Cc1ccccc1", "DAT", 10.0),
        # d-methamphetamine vs l-methamphetamine for DAT
        ("C[C@H](NC)Cc1ccccc1", "C[C@@H](NC)Cc1ccccc1", "DAT", 5.0),
        # S(+)-MDMA vs R(-)-MDMA for SERT (S > R for SERT release)
        ("C[C@H](NC)Cc1ccc2OCOc2c1", "C[C@@H](NC)Cc1ccc2OCOc2c1", "SERT", 3.0),
        # d-amphetamine vs l-amphetamine for NET (d > l)
        ("C[C@H](N)Cc1ccccc1", "C[C@@H](N)Cc1ccccc1", "NET", 3.0),
        # Cocaine - same for both enantiomers (blocker, not substrate - control)
        # Methylphenidate enantiomers for DAT
        ("COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2", "COC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2", "DAT", 5.0),
    ])


@dataclass
class ModelConfig:
    """StereoGNN architecture configuration."""

    # Node (atom) features
    # Actual base atom features: 75 (from featurizer.py)
    # - atomic_num one-hot: 37, degree: 8, formal_charge: 8, num_hs: 6, hybridization: 8
    # - binary features: 6 (aromatic, ring, ring sizes 3-6)
    # - electronegativity: 1, mass: 1
    atom_feature_dim: int = 75  # Base atom features from RDKit
    atom_hidden_dim: int = 128

    # Edge (bond) features
    # Actual base bond features: 11 (from featurizer.py)
    # - bond_type one-hot: 5, conjugated: 1, in_ring: 1, ring sizes 3-6: 4
    bond_feature_dim: int = 11  # Base bond features
    bond_hidden_dim: int = 64

    # Stereochemistry dimensions (added to base features)
    stereo_feature_dim: int = 32  # Combined stereo encoding

    # GNN architecture
    num_gnn_layers: int = 6
    gnn_type: str = "GAT"  # GAT for attention weights interpretability
    num_attention_heads: int = 8

    # Readout
    readout_type: str = "attention"  # attention, mean, max, or set2set
    readout_hidden_dim: int = 256

    # Multi-task heads
    shared_hidden_dim: int = 256
    task_specific_dim: int = 128
    num_tasks: int = 3  # DAT, NET, SERT

    # Regularization
    dropout: float = 0.2
    attention_dropout: float = 0.1

    # Uncertainty quantification
    use_mc_dropout: bool = True
    mc_samples: int = 30

    # Output type
    output_type: str = "classification"  # or "regression" for continuous values
    num_classes: int = 3  # substrate, blocker, inactive


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200

    # Learning rate schedule
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Early stopping
    patience: int = 25
    min_delta: float = 1e-4

    # Loss function - Focal Loss for class imbalance
    loss_type: str = "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Class weights (will be computed from data)
    auto_class_weights: bool = True

    # Multi-task weighting
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "DAT": 1.0,
        "NET": 1.0,
        "SERT": 1.0,
    })

    # Auxiliary losses
    use_bbb_auxiliary: bool = True
    bbb_loss_weight: float = 0.2

    # Data augmentation
    augment_stereo: bool = True  # Enumerate stereoisomers

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # Windows compatibility - multiprocessing can hang


@dataclass
class KineticConfig:
    """Configuration for kinetic parameter prediction extension."""

    # Enable kinetic prediction
    enabled: bool = True

    # Kinetic head architecture
    kinetic_hidden_dim: int = 128
    kinetic_dropout: float = 0.2

    # Output ranges (for normalization/clipping)
    pki_range: Tuple[float, float] = (3.0, 12.0)  # -log10(Ki) typical range
    pic50_range: Tuple[float, float] = (3.0, 12.0)  # -log10(IC50) typical range
    kinetic_bias_range: Tuple[float, float] = (0.0, 1.0)  # Uptake preference

    # Interaction mode classes
    num_interaction_modes: int = 4  # substrate, competitive, non-competitive, partial

    # Uncertainty settings
    min_log_var: float = -10.0
    max_log_var: float = 10.0
    mc_dropout_samples: int = 30

    # Loss weights (if not using learned weights)
    classification_weight: float = 1.0
    pki_weight: float = 1.0
    pic50_weight: float = 1.0
    kinetic_bias_weight: float = 0.5
    mode_weight: float = 1.0

    # Whether to learn task weights via homoscedastic uncertainty
    learn_task_weights: bool = True

    # Data column mappings
    kinetic_columns: Dict[str, str] = field(default_factory=lambda: {
        'pKi': 'pKi',
        'pIC50': 'pIC50',
        'mode': 'interaction_mode',
        'bias': 'kinetic_bias',
        'confidence': 'confidence',
    })

    # Training strategy
    pretrain_classification_first: bool = False  # Joint MAT activity + kinetics training
    pretrain_epochs: int = 50  # Epochs for classification pretraining (unused if joint training)
    freeze_backbone_for_kinetics: bool = False  # Whether to freeze GNN during kinetic training

    # Evaluation thresholds
    min_pki_r2: float = 0.5  # Minimum R² for pKi prediction
    min_pic50_r2: float = 0.5  # Minimum R² for pIC50 prediction
    min_mode_accuracy: float = 0.7  # Minimum accuracy for mode classification


@dataclass
class EvaluationConfig:
    """Evaluation and success criteria configuration."""

    # Primary success thresholds (MUST ACHIEVE)
    min_overall_auroc: float = 0.85
    min_monoamine_auroc: float = 0.95
    min_prauc: float = 0.65
    min_stereo_accuracy: float = 0.80
    min_ablation_drop: float = 0.05

    # Secondary metrics
    enrichment_factors: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10])
    min_ef_1pct: float = 10.0

    # Calibration
    max_ece: float = 0.10  # Expected calibration error

    # Virtual screening validation
    # Known stimulant substrates that MUST rank highly
    known_stimulant_substrates: List[str] = field(default_factory=lambda: [
        "C[C@H](N)Cc1ccccc1",           # d-Amphetamine
        "C[C@H](NC)Cc1ccccc1",          # d-Methamphetamine
        "C[C@H](NC)Cc1ccc2OCOc2c1",     # MDMA
        "NCCc1ccc(O)c(O)c1",            # Dopamine
        "NC[C@H](O)c1ccc(O)c(O)c1",     # Norepinephrine
        "NCCc1c[nH]c2ccc(O)cc12",       # Serotonin
    ])

    # Known blockers that should NOT be predicted as substrates
    known_blockers: List[str] = field(default_factory=lambda: [
        "COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3",  # Cocaine (blocker)
        "c1ccc(C(c2ccccc2)N3CCNCC3)cc1",        # Diphenylmethylpiperazine (blocker)
    ])


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    kinetic: KineticConfig = field(default_factory=KineticConfig)

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.data.models_dir.mkdir(parents=True, exist_ok=True)
        self.data.results_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
CONFIG = Config()


# Known stereoselective pairs for sensitivity testing
# These are compounds where chirality significantly affects transporter activity
STEREOSELECTIVE_PAIRS = [
    {
        'name': 'Amphetamine',
        'd_isomer': 'C[C@H](N)Cc1ccccc1',     # d-Amphetamine (more active)
        'l_isomer': 'C[C@@H](N)Cc1ccccc1',    # l-Amphetamine (less active)
        'target': 'DAT',
        'd_activity': 2,  # Strong substrate
        'l_activity': 0,  # Weak/inactive
    },
    {
        'name': 'Methamphetamine',
        'd_isomer': 'C[C@H](NC)Cc1ccccc1',    # d-Meth (more active)
        'l_isomer': 'C[C@@H](NC)Cc1ccccc1',   # l-Meth (less active)
        'target': 'DAT',
        'd_activity': 2,
        'l_activity': 1,
    },
    {
        'name': 'Amphetamine-NET',
        'd_isomer': 'C[C@H](N)Cc1ccccc1',
        'l_isomer': 'C[C@@H](N)Cc1ccccc1',
        'target': 'NET',
        'd_activity': 2,
        'l_activity': 1,
    },
    {
        'name': 'MDMA',
        'd_isomer': 'C[C@H](NC)Cc1ccc2OCOc2c1',  # S-(+)-MDMA
        'l_isomer': 'C[C@@H](NC)Cc1ccc2OCOc2c1', # R-(-)-MDMA
        'target': 'SERT',
        'd_activity': 2,  # Potent releaser
        'l_activity': 1,  # Less potent
    },
    {
        'name': 'Cathinone',
        'd_isomer': 'C[C@H](N)C(=O)c1ccccc1',    # S-Cathinone (more active)
        'l_isomer': 'C[C@@H](N)C(=O)c1ccccc1',   # R-Cathinone (less active)
        'target': 'DAT',
        'd_activity': 2,
        'l_activity': 1,
    },
    {
        'name': 'Methylphenidate',
        'd_isomer': 'COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2',  # d-threo (Ritalin)
        'l_isomer': 'COC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2',  # l-threo
        'target': 'DAT',
        'd_activity': 1,  # Blocker (not substrate)
        'l_activity': 0,  # Less active
    },
]
