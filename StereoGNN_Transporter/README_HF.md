---
title: StereoGNN Transporter Predictor
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# StereoGNN Transporter Substrate Predictor

Predict monoamine transporter activity for drug molecules using a stereochemistry-aware graph neural network.

## Targets

- **DAT** - Dopamine Transporter
- **NET** - Norepinephrine Transporter
- **SERT** - Serotonin Transporter

## Predictions

| Type | Description |
|------|-------------|
| 🟢 **Substrate** | Actively transported by the transporter |
| 🟡 **Blocker** | Inhibits transporter without being transported |
| ⚪ **Inactive** | No significant interaction |

## Model Performance

| Metric | Value |
|--------|-------|
| Overall ROC-AUC | 0.968 |
| DAT AUC | 0.982 |
| NET AUC | 0.953 |
| SERT AUC | 0.969 |
| Stereo Sensitivity | 83.3% |

## Key Features

- **Stereochemistry-Aware**: Correctly distinguishes between enantiomers (e.g., d- vs l-amphetamine)
- **Multi-Target**: Simultaneous prediction for all three monoamine transporters
- **Interpretable**: Provides substrate probability scores and molecular property analysis

## Example Molecules

| SMILES | Name | Expected |
|--------|------|----------|
| `C[C@H](N)Cc1ccccc1` | d-Amphetamine | DAT/NET substrate |
| `C[C@@H](N)Cc1ccccc1` | l-Amphetamine | Less active |
| `NCCc1ccc(O)c(O)c1` | Dopamine | DAT substrate |
| `NCCc1c[nH]c2ccc(O)cc12` | Serotonin | SERT substrate |

## Usage

Enter a SMILES string in the input box and click "Predict" to get:
1. Transporter predictions with probabilities
2. Molecular structure visualization
3. Molecular properties analysis
4. Interpretation and warnings

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{stereognn_transporter,
  title={StereoGNN: Stereochemistry-Aware Graph Neural Network for Transporter Substrate Prediction},
  year={2024},
  url={https://huggingface.co/spaces/YOUR_USERNAME/stereognn-transporter}
}
```

## Disclaimer

This is a research tool for educational and scientific purposes. Predictions should be validated experimentally before use in any clinical or pharmaceutical context.

## License

MIT License
