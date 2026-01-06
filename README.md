# StereoGNN Drug Discovery Toolkit

A comprehensive in-silico drug discovery platform featuring stereochemistry-aware graph neural networks for ADMET prediction.

## Features

### Monoamine Transporter (MAT) Prediction
- **DAT** - Dopamine Transporter
- **NET** - Norepinephrine Transporter
- **SERT** - Serotonin Transporter

Predictions: Substrate | Blocker | Inactive

### Abuse Liability Prediction
Predicts abuse potential based on MAT activity and structural patterns:
- **HIGH** - Schedule I-II substances (amphetamines, opioids, cocaine)
- **MODERATE** - Schedule III-IV (benzodiazepines, Z-drugs)
- **LOW** - Non-scheduled medications (SSRIs, antipsychotics)

### hERG Cardiotoxicity
Predicts cardiac safety risk via hERG channel inhibition:
- **HIGH** - Significant QT prolongation risk
- **MODERATE** - Some cardiac concerns
- **LOW** - Minimal cardiac liability

### CYP450 Metabolism
Predicts interactions with major CYP enzymes:
- CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4

### Blood-Brain Barrier (BBB) Permeability
Predicts CNS penetration with external validation AUC of 0.9656 on B3DB dataset.

## Model Performance

### MAT Prediction (Internal Validation)
| Metric | Value |
|--------|-------|
| Overall ROC-AUC | 0.968 |
| DAT AUC | 0.982 |
| NET AUC | 0.953 |
| SERT AUC | 0.969 |
| Stereo Sensitivity | 83.3% |

### Abuse Liability (Validation)
| Metric | Value |
|--------|-------|
| Development Set Accuracy | 100% (17 compounds) |
| External Validation | 80 compounds from DEA schedules |

### BBB Prediction (External Validation)
| Metric | Value |
|--------|-------|
| B3DB AUC (7,807 compounds) | 0.9656 |
| Accuracy | 86.65% |
| Sensitivity | 97.88% |

## Key Technologies

- **Stereochemistry-Aware GNN**: Correctly distinguishes enantiomers (d- vs l-amphetamine)
- **Pharmacology Rules Engine**: Post-processing corrections based on known SAR
- **Multi-Task Learning**: Simultaneous prediction across multiple targets
- **Pattern-Based Detection**: SMARTS/SMILES patterns for drug class identification

## Drug Classes Detected

| Class | Detection | Abuse Level |
|-------|-----------|-------------|
| Amphetamines | Phenethylamine + primary amine | HIGH |
| Cathinones | Beta-keto amphetamine | HIGH |
| Opioids | Morphinan/fentanyl scaffolds | HIGH |
| Cocaine-like | Tropane scaffold | HIGH |
| Benzodiazepines | Benzodiazepine core | MODERATE |
| Methylphenidate | Piperidine-phenyl | MODERATE |
| SSRIs | Aryl ether + SERT blocker | LOW |
| SNRIs | SERT + NET blocker | LOW |
| Antipsychotics | Phenothiazine/butyrophenone | LOW |

## Usage

### Web UI (Streamlit)
```bash
cd StereoGNN_Transporter
streamlit run app_ui.py
```

### Python API
```python
from app_ui import predict_molecule, load_models

models = load_models()
results = predict_molecule("CC(N)Cc1ccccc1", models)  # Amphetamine

print(f"DAT: {results['mat']['DAT']['class']}")
print(f"Abuse: {results['abuse_category']}")
print(f"hERG: {results['herg']['risk']}")
```

### Validation Scripts
```bash
# MAT/Abuse/hERG validation (17 known drugs)
python validate_stimulants.py

# External abuse validation (80 compounds)
python external_validation_abuse.py

# BBB external validation (7,807 compounds)
cd ../BBB_System && python external_validation.py
```

## File Structure

```
StereoGNN_Transporter/
├── app_ui.py              # Streamlit web interface
├── app.py                 # Core model definitions
├── model.py               # StereoGNN architecture
├── featurizer.py          # Molecular graph featurization
├── abuse_predictor.py     # Abuse liability prediction
├── pharmacology_rules.py  # SAR-based correction rules
├── validate_stimulants.py # Known drug validation
├── external_validation_abuse.py  # External validation
├── models/                # Trained model weights
│   ├── herg/
│   ├── cyp/
│   └── kinetic_v3/
└── versions/v1/           # Saved model versions
```

## Example Predictions

| Drug | DAT | NET | SERT | Abuse | hERG |
|------|-----|-----|------|-------|------|
| Amphetamine | substrate | substrate | substrate | HIGH | LOW |
| Cocaine | blocker | blocker | blocker | HIGH | MODERATE |
| Methylphenidate | blocker | blocker | inactive | MODERATE | LOW |
| Fluoxetine | inactive | inactive | blocker | LOW | MODERATE |
| Caffeine | inactive | inactive | inactive | LOW | LOW |

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- Streamlit
- scikit-learn
- pandas, numpy

## Citation

```bibtex
@software{stereognn_toolkit,
  title={StereoGNN Drug Discovery Toolkit},
  year={2024},
  note={Stereochemistry-aware ADMET prediction platform}
}
```

## License

MIT License

## Disclaimer

This is a research tool for educational and scientific purposes. All predictions should be validated experimentally before use in clinical or pharmaceutical contexts.
