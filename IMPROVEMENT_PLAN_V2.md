# StereoGNN V2 Improvement Plan
## Target: 99% AUC on Unseen Drugs

**Current State (V1):**
- Validation AUC: 99.2%
- Unseen drug accuracy: 64.7%
- Gap: ~35% accuracy loss on novel scaffolds

---

## Root Cause Analysis

### Why the model fails on unseen drugs:

1. **Training data bias**: ChEMBL data over-represents certain scaffolds (tricyclics, SSRIs, amphetamines)
2. **Substrate/blocker confusion**: Model learns correlations, not mechanisms
3. **SERT under-prediction**: SNRIs/SSRIs in test set have novel features
4. **Blocker over-prediction**: Default when uncertain
5. **Limited chemical diversity**: Training set doesn't cover 2019-2024 drug space

---

## Improvement Strategy (Ordered by Impact)

### Phase 1: Data Augmentation & Expansion (Expected: +15-20%)

#### 1.1 Expand Training Data with Recent Drugs
```
Sources to add:
- DrugBank 2024 (approved drugs with MAT annotations)
- BindingDB (Ki/IC50 data for SLC6A2/3/4)
- PubChem BioAssay (AID 488997, 493208 for DAT/NET/SERT)
- Literature mining for 2020-2024 publications
```

**Target compounds to add:**
| Drug Class | Examples | Expected Count |
|------------|----------|----------------|
| SNRIs | Desvenlafaxine, levomilnacipran, milnacipran | ~50 |
| Novel SSRIs | Vortioxetine, vilazodone | ~30 |
| NRIs | Viloxazine, atomoxetine analogs | ~40 |
| Novel stimulants | Solriamfetol, pitolisant | ~30 |
| Synthetic cathinones | MDPV, α-PHP, N-ethylpentylone | ~100 |

#### 1.2 Stereoisomer Enumeration
```python
# For each training compound with undefined stereo:
# 1. Enumerate all possible stereoisomers
# 2. Assign same label (with uncertainty weight)
# 3. Increases effective training size 2-8x
```

#### 1.3 SMILES Augmentation
```python
# Generate equivalent SMILES representations:
# - Random atom ordering
# - Kekulization variants
# - Tautomer enumeration
# Increases robustness to input variations
```

---

### Phase 2: Architecture Improvements (Expected: +5-10%)

#### 2.1 Upgrade to Larger Model
```python
class StereoGNNv2(nn.Module):
    def __init__(self):
        # Current: 128 hidden, 2 layers, 2 heads
        # V2: 256 hidden, 4 layers, 4 heads
        self.hidden_dim = 256
        self.num_layers = 4
        self.num_heads = 4

        # Add edge attention (currently just features)
        # Add virtual node for global context
        # Add jumping knowledge (concat all layer outputs)
```

#### 2.2 Add Pharmacophore Features
```python
# New atom-level features:
- Distance to nearest basic nitrogen
- Is part of aromatic system connected to amine
- Is alpha to carbonyl (cathinone signature)
- Ring size if in ring
- Is bridgehead atom (tropane signature)

# New global features:
- Lipinski properties (MW, LogP, TPSA, HBD, HBA)
- Number of basic nitrogens
- Phenethylamine substructure count
- Ester/amide count
```

#### 2.3 Multi-Scale Message Passing
```python
# Current: Only 2-hop neighborhood
# V2: Add 3-hop and 4-hop aggregation
# Captures larger pharmacophore patterns
```

---

### Phase 3: Training Improvements (Expected: +5-8%)

#### 3.1 Contrastive Pre-training
```python
# Pre-train on ChEMBL (2M compounds):
# 1. Contrastive learning on molecular similarity
# 2. Predict fingerprint bits as auxiliary task
# 3. Fine-tune on MAT data

# Benefits:
# - Better molecular representations
# - Less overfitting to small MAT dataset
```

#### 3.2 Curriculum Learning
```python
# Train in phases:
# Phase 1: Easy examples (high-confidence labels, common scaffolds)
# Phase 2: Medium (moderate confidence)
# Phase 3: Hard (rare scaffolds, conflicting data)
```

#### 3.3 Label Smoothing & Mixup
```python
# Label smoothing: [1,0,0] -> [0.9, 0.05, 0.05]
# Mixup: Interpolate graphs and labels
# Reduces overconfidence, improves calibration
```

#### 3.4 Class-Balanced Sampling
```python
# Current issue: Inactive >> Blocker >> Substrate
# Solution: Oversample substrates, undersample inactive
# Use sqrt(inverse_frequency) weighting
```

---

### Phase 4: Enhanced Rule Engine (Expected: +5-10%)

#### 4.1 Add More Drug Class Rules

```python
# SNRI Rule (NEW)
SNRI_PATTERNS = {
    'phenylpropylamine': 'CNCCC(*)c1ccccc1',  # Venlafaxine-like
    'cyclopropyl_amine': 'NC1CC1',  # Milnacipran-like
}

IF matches SNRI pattern AND NOT phenethylamine:
    SET SERT = 'blocker'
    SET NET = 'blocker'
    SET DAT = 'inactive'

# SSRI Rule (NEW)
SSRI_PATTERNS = {
    'fluorinated_diphenyl': 'FC(F)(F)c1ccc(*)cc1',
    'benzofuran_indole': 'c1ccc2occc2c1',  # Vilazodone
}

IF matches SSRI pattern AND MW > 300:
    SET SERT = 'blocker'
    SET DAT = 'inactive'
    SET NET = 'inactive'

# Piperidine Stimulant Rule (NEW)
IF has_piperidine AND has_phenyl_attached AND MW < 350:
    SET DAT = 'blocker'
    SET NET = 'blocker'
    # Methylphenidate-like

# NMDA Antagonist Rule (NEW)
NMDA_PATTERNS = {
    'ketamine_core': 'NC1(*)CCCCC1=O',
    'pcp_core': 'C1CCC(N2CCCCC2)CC1',
}

IF matches NMDA pattern:
    SET all transporters = 'inactive'
    # NMDA antagonists don't interact with MATs
```

#### 4.2 Confidence-Weighted Rules
```python
# Only apply rules when model is uncertain
def apply_rules_smart(smiles, predictions):
    for target in ['DAT', 'NET', 'SERT']:
        model_conf = predictions[target]['max_prob']

        if model_conf > 0.85:
            # High confidence - trust model
            continue
        elif model_conf < 0.6:
            # Low confidence - apply rules aggressively
            apply_rules(strong=True)
        else:
            # Medium confidence - apply rules conservatively
            apply_rules(strong=False)
```

#### 4.3 Structural Similarity Fallback
```python
# For unknown scaffolds, find most similar training compound
def similarity_fallback(smiles, training_set):
    fp_query = get_fingerprint(smiles)

    best_sim = 0
    best_match = None
    for train_smiles, train_label in training_set:
        sim = tanimoto(fp_query, get_fingerprint(train_smiles))
        if sim > best_sim:
            best_sim = sim
            best_match = train_label

    if best_sim > 0.7:
        return best_match  # Use similar compound's label
    else:
        return None  # Too novel, use model prediction
```

---

### Phase 5: Ensemble Methods (Expected: +3-5%)

#### 5.1 Model Ensemble
```python
# Train 5 models with different:
# - Random seeds
# - Architecture variants (GATv2, GIN, GraphTransformer)
# - Feature subsets

# Ensemble prediction:
final_pred = weighted_average([
    model1.predict(x),  # GATv2
    model2.predict(x),  # GIN
    model3.predict(x),  # GraphTransformer
    model4.predict(x),  # GATv2 + pharmacophore features
    model5.predict(x),  # Larger model
])
```

#### 5.2 Test-Time Augmentation
```python
# At inference:
# 1. Generate 10 SMILES variants
# 2. Predict each
# 3. Average predictions
# Reduces variance from input representation
```

---

### Phase 6: Active Learning Loop (Expected: +5-10%)

```python
# Identify high-uncertainty predictions
uncertain_compounds = []
for smiles in drug_database:
    pred = model.predict(smiles)
    entropy = -sum(p * log(p) for p in pred['probs'])
    if entropy > threshold:
        uncertain_compounds.append(smiles)

# Prioritize for experimental validation
# Add validated labels to training set
# Retrain model
# Repeat
```

---

## Implementation Roadmap

| Phase | Task | Time | Expected Gain |
|-------|------|------|---------------|
| 1.1 | Expand training data | 1 week | +10% |
| 1.2 | Stereoisomer enumeration | 2 days | +3% |
| 1.3 | SMILES augmentation | 1 day | +2% |
| 2.1 | Larger model | 2 days | +3% |
| 2.2 | Pharmacophore features | 3 days | +4% |
| 3.1 | Contrastive pretraining | 1 week | +5% |
| 3.2-3.4 | Training improvements | 3 days | +3% |
| 4.1-4.3 | Enhanced rules | 2 days | +5% |
| 5.1-5.2 | Ensemble | 3 days | +3% |
| **Total** | | **~4 weeks** | **+35-40%** |

**Target: 64.7% + 35% = ~99% on unseen drugs**

---

## Quick Wins (Implement First)

### 1. Add SNRI/SSRI Rules (2 hours)
Most impactful for current failures (desvenlafaxine, vortioxetine, vilazodone)

### 2. Confidence-Weighted Rule Application (1 hour)
Only override low-confidence predictions

### 3. Expand Training with BindingDB Data (1 day)
Add ~500 more compounds with Ki data

### 4. Pharmacophore Features (4 hours)
Add basic nitrogen count, phenethylamine flag to global features

---

## Validation Strategy

After each phase, validate on:
1. **Original validation set** (should maintain 99% AUC)
2. **Unseen drug set** (target: improve from 64.7%)
3. **Stimulant set** (should maintain/improve)
4. **New hold-out set** (create from BindingDB)

Track:
- Overall accuracy
- Per-class accuracy (substrate vs blocker vs inactive)
- Per-target accuracy (DAT vs NET vs SERT)
- Calibration (predicted probability vs actual accuracy)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting to expanded data | Use stratified splits, monitor validation |
| Rules become too specific | Validate rules on diverse test sets |
| Ensemble too slow | Use knowledge distillation |
| New drugs still fail | Maintain "uncertainty" flag in output |

---

## Success Criteria

**V2 is successful if:**
1. Unseen drug accuracy > 85%
2. No regression on validation AUC (maintain > 99%)
3. Stimulant accuracy > 80%
4. SNRI/SSRI accuracy > 75%
5. Cathinone accuracy = 100%
