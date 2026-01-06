# Morning Checklist - Training Pipeline

## When You Wake Up

### 1. Check hERG Training Results
```bash
cd C:\Users\nakhi\StereoGNN_Transporter
type models\herg\results.json
```
Expected: ~0.90+ AUC with FP fusion

### 2. Pull Extra Cardiac Data (if not done)
```bash
python pull_herg_extra.py
```
This gets: TDC, PubChem, Nav1.5, Cav1.2, QT data

### 3. Train CYP Model
```bash
python train_cyp.py
```
~38k molecules, 5 targets (1A2, 2C9, 2C19, 2D6, 3A4)
Takes ~2-3 hours

### 4. Train Full Cardiac Panel (optional)
```bash
python train_cardiac_panel.py
```
Multi-task: hERG + Nav1.5 + Cav1.2

### 5. Test Unified Predictor
```bash
python predict_toolkit.py "CC(N)Cc1ccccc1"
```
Should run all models on amphetamine

---

## Models Status

| Model | Script | Status |
|-------|--------|--------|
| BBB | (existing) | ✅ Done |
| MAT Kinetics | train_kinetic_v3.py | ✅ Done (0.90 AUC) |
| Abuse Liability | abuse_liability.py | ✅ Done |
| hERG | train_herg.py | 🔄 Training overnight |
| CYP450 | train_cyp.py | ⏳ Ready to run |
| Cardiac Panel | train_cardiac_panel.py | ⏳ Ready to run |

---

## Files Created Tonight

- `train_herg.py` - hERG with FP fusion
- `train_cyp.py` - Multi-task CYP model
- `train_cardiac_panel.py` - Full CiPA panel
- `pull_herg_extra.py` - Extra data sources
- `pull_herg_clean.py` - Clean IC50 data
- `predict_toolkit.py` - Unified inference API

---

## Deployment Ready After:
1. hERG finishes ✓
2. CYP trains ✓
3. (Optional) Cardiac panel

Then `predict_toolkit.py` gives you one-line predictions for everything.
