# Insilico Drug Discovery Toolkit - User Guide

## Quick Start

1. **Launch the UI:**
   ```bash
   cd C:\Users\nakhi\StereoGNN_Transporter
   streamlit run app_ui.py
   ```

2. **Enter a compound:** Type either:
   - A drug name (e.g., `cocaine`, `fluoxetine`, `atorvastatin`)
   - A SMILES string (e.g., `CC(N)Cc1ccccc1`)

3. **View predictions** for:
   - hERG cardiotoxicity
   - CYP450 metabolism (1A2, 2C9, 2C19, 2D6, 3A4)
   - MAT transporter kinetics (DAT, NET, SERT)
   - Abuse liability score

---

## Modules & Performance

| Module | What it Predicts | AUC |
|--------|------------------|-----|
| **hERG** | Cardiac ion channel blockade (QT prolongation risk) | 0.91 |
| **CYP450** | Drug metabolism enzyme inhibition | 0.88 |
| **MAT Kinetics** | Monoamine transporter activity (DAT/NET/SERT) | 0.90 |
| **Abuse Liability** | Addiction potential based on DAT profile | Rule-based |

---

## Important Caveats & Limitations

### 1. Prodrugs
**The model predicts based on molecular structure, not metabolism.**

- **Lisdexamphetamine** (Vyvanse) is a prodrug of dexamphetamine
- The model sees them as different molecules and may give different predictions
- **For prodrugs, predict the active metabolite for clinical relevance**

Common prodrugs to be aware of:
| Prodrug | Active Form |
|---------|-------------|
| Lisdexamphetamine | Dexamphetamine |
| Codeine | Morphine (via CYP2D6) |
| Clopidogrel | Active thiol metabolite |
| Enalapril | Enalaprilat |
| Valacyclovir | Acyclovir |

### 2. Stereochemistry
- The model is stereo-aware but PubChem may return racemic SMILES
- For chiral drugs, the specific enantiomer matters (e.g., S-ketamine vs R-ketamine)
- If stereochemistry is critical, input the specific isomer's SMILES

### 3. Concentration/Dose Not Considered
- Predictions are binary (active/inactive) or probability-based
- Real toxicity depends on **dose and plasma concentration**
- A drug predicted as "hERG blocker" may be safe at therapeutic doses
- Example: Many antihistamines block hERG but are safe at normal doses

### 4. Multi-Drug Interactions
- The model predicts single compounds only
- Drug-drug interactions (DDIs) are not captured
- CYP inhibition predictions hint at DDI potential but don't model combinations

### 5. Metabolites
- The model predicts parent compounds only
- Some drugs have active/toxic metabolites not captured here
- Example: Acetaminophen → NAPQI (hepatotoxic metabolite)

### 6. Species Differences
- Models trained on human data
- Not validated for veterinary or cross-species predictions

### 7. Novel Scaffolds
- Performance may degrade on chemical structures very different from training data
- Best accuracy on drug-like molecules (Lipinski-compliant)

---

## Interpreting Results

### hERG Cardiotoxicity
| Probability | Risk Level | Interpretation |
|-------------|------------|----------------|
| < 0.3 | LOW | Likely safe for cardiac effects |
| 0.3 - 0.7 | MODERATE | Further testing recommended |
| > 0.7 | HIGH | Significant hERG liability - caution |

### CYP450 Inhibition
- **Inhibitor = YES**: May cause drug-drug interactions by blocking metabolism
- **Clinical significance** depends on whether victim drugs use that CYP
- CYP3A4 inhibition is most clinically relevant (metabolizes ~50% of drugs)

### Abuse Liability Score
| Score | Category | Interpretation |
|-------|----------|----------------|
| 0-35 | LOW | Minimal abuse concern |
| 35-60 | MODERATE | Some reinforcing properties |
| 60-100 | HIGH | Significant abuse potential |

Key factors:
- DAT substrate (releaser) = highest risk (amphetamine-like)
- DAT blocker = high risk (cocaine-like)
- SERT activity reduces abuse potential (dysphoric effects)

---

## Programmatic Usage

```python
from predict_toolkit import DrugDiscoveryToolkit

toolkit = DrugDiscoveryToolkit()

# Single prediction
result = toolkit.predict("CC(N)Cc1ccccc1")  # Amphetamine
print(result.summary())

# Batch prediction
results = toolkit.predict_batch(["CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"])

# CSV processing
toolkit.predict_csv("input.csv", "output.csv", smiles_col="SMILES")
```

---

## Troubleshooting

### "Invalid SMILES" error
- Check SMILES syntax (balanced parentheses, valid atoms)
- Try the drug name instead - PubChem lookup handles conversion

### Drug name not found
- Check spelling
- Try generic name instead of brand name
- Very new drugs may not be in PubChem yet

### Model not loading
- Ensure model files exist in `models/` directory
- Check: `models/herg/best_fold0.pt`, `models/cyp/best_cyp_model.pt`

---

## Citation

If using this toolkit in research, please cite:
- StereoGNN architecture for stereo-aware molecular learning
- Training data sources: ChEMBL, TDC, PubChem BioAssay

---

*Last updated: January 2026*
