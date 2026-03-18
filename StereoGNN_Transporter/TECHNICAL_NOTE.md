# StereoGNN Technical Note: Stereochemistry-Aware Graph Neural Network for Monoamine Transporter Substrate Prediction

## Executive Summary

StereoGNN is a stereochemistry-aware graph neural network designed to predict whether small molecules act as substrates, blockers, or inactive compounds at monoamine transporters (DAT, NET, SERT). The key innovation is explicit encoding of stereochemical information, which is critical for distinguishing pharmacologically active enantiomers.

**Key Results (Targets)**:
- Overall ROC-AUC: ≥0.85
- Monoamine-specific ROC-AUC: ≥0.95
- Substrate PR-AUC: ≥0.65
- Stereo sensitivity: ≥80% on known enantiomer pairs
- Ablation drop: ≥5% when removing stereo features

## 1. Problem Statement

### 1.1 Clinical Relevance

Monoamine transporters (DAT, NET, SERT) are critical targets for:
- **Antidepressants**: SSRIs, SNRIs (blockers)
- **ADHD medications**: Amphetamines, methylphenidate (substrates/releasers)
- **Drugs of abuse**: Cocaine (blocker), methamphetamine (substrate)
- **Novel psychoactive substances (NPS)**: Synthetic cathinones, novel amphetamines

### 1.2 The Substrate vs Blocker Distinction

| Property | Substrate | Blocker |
|----------|-----------|---------|
| **Mechanism** | Transported into cell, causes neurotransmitter release | Binds but not transported |
| **Effect** | Releasing agent (↑↑ neurotransmitter) | Reuptake inhibitor (↑ neurotransmitter) |
| **Examples** | Amphetamine, MDMA | Cocaine, fluoxetine |
| **Abuse potential** | Higher (euphoria, reinforcement) | Lower (but still significant) |

### 1.3 Why Stereochemistry Matters

Enantiomers can have dramatically different activities:

| Compound | d-isomer | l-isomer | Ratio |
|----------|----------|----------|-------|
| Amphetamine (DAT) | High substrate | Low substrate | ~10:1 |
| Methamphetamine (DAT) | High substrate | Lower substrate | ~5:1 |
| MDMA (SERT) | Potent releaser | Less potent | ~2:1 |
| Modafinil | DAT blocker | Less active | Variable |

**Implication**: A model that ignores stereochemistry will fail on clinically relevant distinctions.

## 2. Architecture

### 2.1 Two-Stage Training Strategy

```
Stage 1: Pretraining (General Transporter Capability)
├── All SLC transporters (SLC6, SLC22, ABC, etc.)
├── 50k-100k compounds
├── Learn general "transporter substrate" features:
│   ├── Amphiphilicity
│   ├── Cationic/zwitterionic character
│   ├── Molecular size constraints
│   └── Lipophilicity patterns

Stage 2: Fine-tuning (Monoamine Specialization)
├── DAT/NET/SERT specific data
├── ~1000-2000 compounds (with augmentation)
├── Learn monoamine-specific features:
│   ├── Phenethylamine scaffold recognition
│   ├── α-carbon chirality sensitivity
│   └── Target selectivity patterns
```

### 2.2 Model Architecture

```
Input: Molecular Graph (SMILES → 2D graph)
         │
         ▼
┌─────────────────────────────────────────────┐
│         Stereo-Aware Node Encoder           │
│  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Base Features│  │ Stereo Features     │   │
│  │ (78 dims)    │  │ (11 dims):          │   │
│  │ - Atom type  │  │ - R/S config (2)    │   │
│  │ - Degree     │  │ - Chiral tags (9)   │   │
│  │ - Hybridiz.  │  │                     │   │
│  └──────┬───────┘  └─────────┬───────────┘   │
│         │                     │              │
│         ▼                     ▼              │
│    ┌────────┐           ┌──────────┐         │
│    │ MLP    │           │ Stereo   │         │
│    │ Encoder│           │ Encoder  │         │
│    └────┬───┘           └────┬─────┘         │
│         │                    │               │
│         ├────────────────────┤               │
│         ▼                    │               │
│    ┌───────────────┐    ┌────┴────┐          │
│    │ Fusion Layer  │◄───┤ Stereo  │          │
│    │               │    │  Gate   │          │
│    └───────┬───────┘    └─────────┘          │
└────────────┼────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│      GATv2 Message Passing (4 layers)        │
│  - Multi-head attention (4 heads)            │
│  - Edge-aware convolutions                   │
│  - Residual connections                      │
│  - LayerNorm                                 │
└─────────────┬───────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│         Attention Readout                    │
│  - Graph-level pooling with learned weights  │
│  - Node importance for interpretability      │
└─────────────┬───────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│         Shared Representation                │
│  - 256-dim hidden layer                      │
│  - LayerNorm + Dropout                       │
└─────────────┬───────────────────────────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
┌────────┐ ┌────────┐ ┌────────┐
│  DAT   │ │  NET   │ │  SERT  │
│  Head  │ │  Head  │ │  Head  │
└────┬───┘ └────┬───┘ └────┬───┘
     │          │          │
     ▼          ▼          ▼
[inactive, blocker, substrate] × 3 targets
```

### 2.3 Stereochemistry Encoding

**Node-level stereo features (11 dimensions)**:
```python
stereo_features = [
    one_hot(chiral_tag, 9),  # CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW,
                              # CHI_TETRAHEDRAL_CCW, CHI_OTHER, etc.
    is_R_stereocenter,        # R configuration
    is_S_stereocenter,        # S configuration
    is_chiral_center,         # Has defined chirality
]
```

**Edge-level stereo features (7 dimensions)**:
```python
stereo_features = [
    one_hot(bond_stereo, 6),  # STEREONONE, STEREOANY, STEREOZ,
                               # STEREOE, STEREOCIS, STEREOTRANS
    is_stereo_bond,           # Has defined E/Z geometry
]
```

**Key insight**: The stereo gate learns to upweight stereochemistry features when chiral centers are present, and downweight them for achiral molecules.

## 3. Data Curation

### 3.1 Data Sources

| Source | Compounds | Notes |
|--------|-----------|-------|
| Literature (curated) | ~200 | High-confidence, known substrates/blockers |
| SAR expansion | ~200 | Systematic analogs of known compounds |
| ChEMBL (filtered) | ~100+ | Strict assay-type filtering |
| Decoys | ~50 | Inactive controls (druglike, no activity) |
| **Total (pre-augment)** | **~500** | |
| **With stereo augmentation** | **~650+** | Enumerated stereoisomers |

### 3.2 Substrate vs Blocker Classification

Critical distinction based on **assay type**:

**Substrate assays** (label = 2):
- Uptake/release assays (superfusion, efflux)
- Transport assays (Km, Vmax)
- Keywords: "release", "releasing", "efflux", "substrate"

**Blocker assays** (label = 1):
- Binding assays (radioligand displacement)
- Uptake inhibition (IC50, Ki)
- Keywords: "inhibitor", "inhibition", "binding", "antagonist"

**Inactive** (label = 0):
- Tested but no activity (pChEMBL < 5)
- Decoy compounds (structurally dissimilar, no expected activity)

### 3.3 Stereoisomer Augmentation

For compounds without defined stereochemistry:
1. Enumerate all possible stereoisomers (max 8)
2. Propagate the label to all isomers
3. Mark as augmented (for confidence weighting)

**Rationale**: If a racemic mixture is active, at least one stereoisomer must be active.

## 4. Training Details

### 4.1 Loss Function

**Focal Loss** for class imbalance:
```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
```
- γ = 2 (focus on hard examples)
- Class weights computed from inverse frequency

**Multi-task Loss**:
```
L_total = Σ_t w_t × L_t
```
- Learned task weights (uncertainty weighting)
- Gradients balanced across DAT/NET/SERT

### 4.2 Uncertainty Quantification

**MC Dropout** (30 forward passes):
- Keep dropout active during inference
- Compute mean and std of predictions
- High uncertainty → flag for human review

### 4.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (pretrain), 1e-5 (fine-tune backbone), 1e-4 (fine-tune heads) |
| Batch size | 32 |
| Epochs (pretrain) | 100 |
| Epochs (fine-tune) | 50 |
| Early stopping | 10 epochs patience |
| Gradient clipping | 1.0 |
| Dropout | 0.1 (GNN), 0.2 (heads) |

## 5. Evaluation

### 5.1 Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Overall ROC-AUC | ≥0.85 | Competitive with existing tools |
| Monoamine ROC-AUC | ≥0.95 | Specialized performance |
| Substrate PR-AUC | ≥0.65 | Overcome class imbalance |
| Stereo sensitivity | ≥80% | Distinguish enantiomers |
| Ablation drop | ≥5% | Prove stereo features matter |

### 5.2 Stereo Sensitivity Test

Test on known stereoselective pairs:

| Pair | Target | d-activity | l-activity | Expected |
|------|--------|------------|------------|----------|
| Amphetamine | DAT | Substrate | Inactive | P(d) > P(l) |
| Methamphetamine | DAT | Substrate | Lower | P(d) > P(l) |
| MDMA | SERT | Substrate | Lower | P(d) > P(l) |

**Pass criterion**: Model correctly predicts P(substrate | d) > P(substrate | l) for ≥80% of pairs.

### 5.3 Ablation Study

Compare full model vs model without stereo features:

| Model | Features | Expected AUC drop |
|-------|----------|-------------------|
| Full StereoGNN | All | Baseline |
| No stereo node | Remove R/S encoding | 2-3% |
| No stereo edge | Remove E/Z encoding | 1-2% |
| No stereo at all | Base features only | ≥5% |

## 6. Interpretability

### 6.1 Attention Weights

The attention readout layer provides atom-level importance:
- **High attention**: Atoms critical for prediction
- For substrates: α-carbon, amine nitrogen typically highlighted
- For blockers: Binding pharmacophore highlighted

### 6.2 Embedding Visualization

t-SNE/UMAP of learned embeddings should show:
- Clear separation between substrates/blockers/inactive
- Enantiomers separated (not overlapping)
- Target-specific clustering (DAT actives near each other)

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Data scarcity**: ~500-650 compounds is small compared to other molecular property datasets
2. **Label ambiguity**: Some compounds are both substrates AND blockers (partial substrates)
3. **Selectivity**: Model predicts per-target, but some compounds have complex selectivity profiles
4. **3D conformations**: Currently using 2D graphs; 3D could improve stereo encoding

### 7.2 Future Directions

1. **3D StereoGNN**: Incorporate 3D coordinates for explicit spatial chirality
2. **Active learning**: Prioritize compounds for experimental validation
3. **Multi-target selectivity**: Predict DAT/NET/SERT ratios jointly
4. **Partial substrate detection**: Continuous substrate/blocker spectrum
5. **Transporter kinetics**: Predict Km, Vmax, not just classification

## 8. Usage

### 8.1 Training

```bash
cd StereoGNN_Transporter

# Prepare data
python verify_data_final.py

# Train (pretrain + fine-tune)
python train_pipeline.py

# Evaluate
python evaluation.py --checkpoint outputs/finetune/best_finetune.pt
```

### 8.2 Inference

```python
from inference import StereoGNNPredictor

predictor = StereoGNNPredictor("outputs/finetune/best_finetune.pt")

# Single prediction
result = predictor.predict("C[C@H](N)Cc1ccccc1")  # d-Amphetamine
print(result)
# {'DAT': {'substrate': 0.95, 'blocker': 0.03, 'inactive': 0.02, 'uncertainty': 0.02},
#  'NET': {'substrate': 0.88, 'blocker': 0.08, 'inactive': 0.04, 'uncertainty': 0.03},
#  'SERT': {'substrate': 0.45, 'blocker': 0.35, 'inactive': 0.20, 'uncertainty': 0.05}}

# Compare enantiomers
d_amp = predictor.predict("C[C@H](N)Cc1ccccc1")
l_amp = predictor.predict("C[C@@H](N)Cc1ccccc1")

print(f"d-Amphetamine DAT substrate: {d_amp['DAT']['substrate']:.2f}")
print(f"l-Amphetamine DAT substrate: {l_amp['DAT']['substrate']:.2f}")
# d-Amphetamine DAT substrate: 0.95
# l-Amphetamine DAT substrate: 0.25
```

## 9. References

1. Sitte HH, Freissmuth M. (2015). Amphetamines, new psychoactive drugs and the monoamine transporter cycle. Trends Pharmacol Sci.
2. Rothman RB, Baumann MH. (2003). Monoamine transporters and psychostimulant drugs. Eur J Pharmacol.
3. Gilson MK et al. (2016). BindingDB: Making experimental data machine actionable. Nucleic Acids Res.
4. Mendez D et al. (2019). ChEMBL: towards direct deposition of bioassay data. Nucleic Acids Res.

---

**Author**: StereoGNN Development Team
**Version**: 1.0
**Date**: 2024
