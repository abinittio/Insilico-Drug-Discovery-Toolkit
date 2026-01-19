---
title: Insilico Drug Discovery Toolkit
emoji: 💊
colorFrom: teal
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app_ui.py
pinned: true
license: mit
---

# Insilico Drug Discovery Toolkit

Comprehensive AI-powered ADMET prediction suite for drug discovery.

## Prediction Models

| Model | Target | Performance |
|-------|--------|-------------|
| **MAT Transporter** | DAT/NET/SERT substrate/blocker | 0.968 AUC |
| **Abuse Liability** | HIGH/MODERATE/LOW scoring | 100% Dev Accuracy |
| **hERG Cardiotoxicity** | Cardiac safety | 0.91 AUC |
| **CYP450 Metabolism** | Drug-drug interactions | 0.88 AUC |

## Features

- **Multi-Target Predictions**: Simultaneous ADMET predictions
- **Stereochemistry-Aware**: Distinguishes between enantiomers
- **Drug Class Detection**: Identifies amphetamines, opioids, benzodiazepines, etc.
- **Pharmacology Rules**: SAR-based post-processing corrections
- **Batch Processing**: Upload CSV files for bulk predictions

## Drug Classes Detected

- Amphetamines, Cathinones (HIGH abuse potential)
- Opioids, Cocaine-like (HIGH abuse potential)
- Benzodiazepines, Z-drugs (MODERATE abuse potential)
- SSRIs, SNRIs, Antipsychotics (LOW abuse potential)

## Usage

1. Enter SMILES or drug name
2. Select prediction type (MAT, Abuse, hERG, CYP)
3. View results with molecular visualization
4. Export predictions

## Author

**Nabil Yasini-Ardekani**
[GitHub](https://github.com/abinittio) | [Dis-Solved](https://dis-solved.com)

## License

MIT License
