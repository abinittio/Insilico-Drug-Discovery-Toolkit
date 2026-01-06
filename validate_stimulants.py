"""
Stimulant & ADHD Medication Validation Suite
=============================================
Systematic testing of model predictions against known pharmacology.

Run: python validate_stimulants.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

# Add parent dir
sys.path.insert(0, str(Path(__file__).parent))

import torch
from rdkit import Chem

# Known pharmacology from literature
# Sources: Goodman & Gilman, PubChem, DrugBank, Primary literature

STIMULANT_PHARMACOLOGY = {
    # ===================
    # AMPHETAMINES (DAT/NET releasers)
    # ===================
    "amphetamine": {
        "smiles": "CC(N)Cc1ccccc1",
        "class": "Amphetamine",
        "expected_dat": "substrate",  # Releases dopamine
        "expected_net": "substrate",  # Releases norepinephrine
        "expected_sert": "substrate",  # Weak serotonin release
        "expected_abuse": "HIGH",  # Schedule II
        "expected_herg": "LOW",  # Not cardiotoxic at therapeutic doses
        "notes": "Prototypical psychostimulant, DAT/NET releaser"
    },
    "dexamphetamine": {
        "smiles": "C[C@H](N)Cc1ccccc1",
        "class": "Amphetamine",
        "expected_dat": "substrate",
        "expected_net": "substrate",
        "expected_sert": "substrate",
        "expected_abuse": "HIGH",
        "expected_herg": "LOW",
        "notes": "D-enantiomer, more potent than L-amphetamine"
    },
    "levoamphetamine": {
        "smiles": "C[C@@H](N)Cc1ccccc1",
        "class": "Amphetamine",
        "expected_dat": "substrate",
        "expected_net": "substrate",
        "expected_sert": "inactive",
        "expected_abuse": "MODERATE",  # Less potent than D-isomer
        "expected_herg": "LOW",
        "notes": "L-enantiomer, more peripheral effects"
    },
    "methamphetamine": {
        "smiles": "CC(Cc1ccccc1)NC",
        "class": "Amphetamine",
        "expected_dat": "substrate",
        "expected_net": "substrate",
        "expected_sert": "substrate",
        "expected_abuse": "HIGH",  # Schedule II, high abuse
        "expected_herg": "LOW",
        "notes": "N-methyl amphetamine, more lipophilic, crosses BBB faster"
    },
    "lisdexamfetamine": {
        "smiles": None,  # Will fetch from PubChem
        "class": "Amphetamine prodrug",
        "expected_dat": "inactive",  # PRODRUG - not active until cleaved
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "expected_abuse": "MODERATE",  # Lower due to prodrug mechanism
        "expected_herg": "LOW",
        "notes": "PRODRUG of dexamphetamine - model may not capture this"
    },
    "mdma": {
        "smiles": "CC(N)Cc1ccc2OCOc2c1",
        "class": "Amphetamine",
        "expected_dat": "substrate",
        "expected_net": "substrate",
        "expected_sert": "substrate",  # STRONG serotonin releaser
        "expected_abuse": "HIGH",
        "expected_herg": "MODERATE",  # Some cardiotoxicity reports
        "notes": "Ecstasy - potent SERT releaser, entactogen"
    },

    # ===================
    # METHYLPHENIDATE CLASS (DAT/NET blockers)
    # ===================
    "methylphenidate": {
        "smiles": "COC(=O)C(c1ccccc1)C1CCCCN1",
        "class": "Methylphenidate",
        "expected_dat": "blocker",  # Blocks reuptake
        "expected_net": "blocker",
        "expected_sert": "inactive",  # Minimal SERT
        "expected_abuse": "MODERATE",  # Schedule II but lower than amphetamines
        "expected_herg": "LOW",
        "notes": "Ritalin - DAT/NET blocker, not releaser"
    },
    "dexmethylphenidate": {
        "smiles": None,  # Will fetch
        "class": "Methylphenidate",
        "expected_dat": "blocker",
        "expected_net": "blocker",
        "expected_sert": "inactive",
        "expected_abuse": "MODERATE",
        "expected_herg": "LOW",
        "notes": "D-threo enantiomer, active form of methylphenidate"
    },

    # ===================
    # OTHER ADHD MEDICATIONS
    # ===================
    "atomoxetine": {
        "smiles": None,
        "class": "NRI",
        "expected_dat": "inactive",  # Minimal DAT
        "expected_net": "blocker",   # Selective NET inhibitor
        "expected_sert": "inactive",
        "expected_abuse": "LOW",     # Not scheduled, non-stimulant
        "expected_herg": "LOW",
        "notes": "Strattera - selective NET inhibitor, non-stimulant ADHD med"
    },
    "modafinil": {
        "smiles": "NC(=O)CS(=O)C(c1ccccc1)c1ccccc1",
        "class": "Eugeroic",
        "expected_dat": "blocker",  # Weak DAT blocker
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "expected_abuse": "LOW",    # Schedule IV
        "expected_herg": "LOW",
        "notes": "Provigil - weak DAT blocker, wakefulness promoter"
    },
    "bupropion": {
        "smiles": "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1",
        "class": "NDRI",
        "expected_dat": "blocker",  # Weak DAT blocker
        "expected_net": "blocker",  # Weak NET blocker
        "expected_sert": "inactive",
        "expected_abuse": "LOW",    # Not scheduled
        "expected_herg": "LOW",
        "notes": "Wellbutrin - weak NDRI, antidepressant, smoking cessation"
    },

    # ===================
    # COMPARISON DRUGS
    # ===================
    "cocaine": {
        "smiles": "COC(=O)C1C2CCC(CC1OC(=O)c1ccccc1)N2C",
        "class": "Tropane",
        "expected_dat": "blocker",
        "expected_net": "blocker",
        "expected_sert": "blocker",
        "expected_abuse": "HIGH",   # Schedule II
        "expected_herg": "MODERATE",  # Known cardiotoxicity
        "notes": "Triple reuptake inhibitor, local anesthetic, high abuse"
    },
    "caffeine": {
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "class": "Xanthine",
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "expected_abuse": "LOW",
        "expected_herg": "LOW",
        "notes": "Adenosine antagonist, not a monoamine drug"
    },
    "nicotine": {
        "smiles": "CN1CCCC1c1cccnc1",
        "class": "Alkaloid",
        "expected_dat": "inactive",  # Indirect dopamine release via nAChR
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "expected_abuse": "HIGH",    # Highly addictive
        "expected_herg": "LOW",
        "notes": "nAChR agonist, indirect dopamine effects"
    },

    # ===================
    # NEGATIVE CONTROLS (non-stimulants)
    # ===================
    "fluoxetine": {
        "smiles": None,
        "class": "SSRI",
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "blocker",  # Selective SERT inhibitor
        "expected_abuse": "LOW",
        "expected_herg": "MODERATE",  # Some QT concerns
        "notes": "Prozac - SSRI, serotonin selective"
    },
    "venlafaxine": {
        "smiles": None,
        "class": "SNRI",
        "expected_dat": "inactive",
        "expected_net": "blocker",
        "expected_sert": "blocker",
        "expected_abuse": "LOW",
        "expected_herg": "MODERATE",
        "notes": "Effexor - SNRI, dose-dependent NET effects"
    },
    "haloperidol": {
        "smiles": None,
        "class": "Antipsychotic",
        "expected_dat": "blocker",  # D2 antagonist
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "expected_abuse": "LOW",
        "expected_herg": "HIGH",    # Known QT prolongation
        "notes": "D2 antagonist, known hERG liability"
    },
}


def fetch_smiles(name: str) -> str:
    """Fetch SMILES from PubChem."""
    import requests
    import urllib.parse

    try:
        encoded = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data['PropertyTable']['Properties'][0].get('SMILES')
    except:
        pass
    return None


def load_models():
    """Load all prediction models."""
    from model import StereoGNN
    from featurizer import MoleculeGraphFeaturizer

    models = {}
    model_dir = Path(__file__).parent / 'models'

    # hERG
    herg_path = model_dir / 'herg' / 'best_fold0.pt'
    if herg_path.exists():
        from train_herg import HERGClassifier, FP_DIM
        backbone = StereoGNN()
        models['herg'] = HERGClassifier(backbone, fp_dim=FP_DIM)
        ckpt = torch.load(herg_path, map_location='cpu', weights_only=False)
        models['herg'].load_state_dict(ckpt['model_state_dict'])
        models['herg'].eval()
        print("  [OK] hERG model loaded")

    # CYP
    cyp_path = model_dir / 'cyp' / 'best_cyp_model.pt'
    if cyp_path.exists():
        from train_cyp import CYPClassifier, CYP_TARGETS, FP_DIM
        backbone = StereoGNN()
        models['cyp'] = CYPClassifier(backbone, fp_dim=FP_DIM, n_targets=len(CYP_TARGETS))
        ckpt = torch.load(cyp_path, map_location='cpu', weights_only=False)
        models['cyp'].load_state_dict(ckpt['model_state_dict'])
        models['cyp'].eval()
        models['cyp_targets'] = CYP_TARGETS
        print("  [OK] CYP model loaded")

    # MAT - Original SOTA model (0.99 AUC)
    mat_path = Path(__file__).parent / 'best_model.pt'
    if mat_path.exists():
        from app import StereoGNNSmallFinetune
        models['mat'] = StereoGNNSmallFinetune()
        ckpt = torch.load(mat_path, map_location='cpu', weights_only=False)
        models['mat'].load_state_dict(ckpt['model_state_dict'])
        models['mat'].eval()
        print(f"  [OK] MAT model loaded (AUC: {ckpt.get('best_auc', 0):.3f})")

    models['featurizer'] = MoleculeGraphFeaturizer(use_3d=False)

    return models


def get_fingerprint(smiles: str):
    """Generate fingerprint for model."""
    from rdkit.Chem import AllChem, Descriptors, DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_arr = np.zeros(1024)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    desc = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolWt(mol) / 500,
        Descriptors.NumHDonors(mol) / 5,
        Descriptors.NumHAcceptors(mol) / 10,
        Descriptors.NumRotatableBonds(mol) / 10,
        Descriptors.NumAromaticRings(mol) / 5,
        Descriptors.FractionCSP3(mol),
    ]

    return np.concatenate([fp_arr, desc]).astype(np.float32)


@torch.no_grad()
def predict_compound(smiles: str, models: dict) -> dict:
    """Run all predictions on a compound."""
    result = {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': 'Invalid SMILES'}

    # Featurize
    data = models['featurizer'].featurize(smiles, {'DAT': -1, 'NET': -1, 'SERT': -1})
    if data is None:
        return {'error': 'Featurization failed'}

    fp = get_fingerprint(smiles)
    if fp is not None:
        data.fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)

    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data])

    # hERG
    if 'herg' in models:
        output = models['herg'](batch)
        prob = torch.softmax(output['logits'], dim=-1)[0, 1].item()
        result['herg_prob'] = prob
        result['herg_risk'] = 'HIGH' if prob > 0.7 else ('MODERATE' if prob > 0.3 else 'LOW')

    # MAT (original SOTA model)
    if 'mat' in models:
        mat_output = models['mat'](batch)

        for target in ['DAT', 'NET', 'SERT']:
            logits = mat_output[target][0]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax().item()
            pred_class = ['inactive', 'blocker', 'substrate'][pred_idx]
            result[f'{target}_class'] = pred_class
            result[f'{target}_probs'] = probs.numpy()

        # Apply pharmacology rules to correct MAT predictions
        from pharmacology_rules import PharmacologyRules
        rules = PharmacologyRules()

        # Prepare predictions dict for rule correction
        raw_preds = {
            'DAT_class': result['DAT_class'],
            'NET_class': result['NET_class'],
            'SERT_class': result['SERT_class'],
            'herg_risk': result.get('herg_risk'),
        }

        # Apply corrections
        corrected = rules.correct_predictions(smiles, raw_preds)

        # Update result with corrected values
        result['DAT_class'] = corrected.get('DAT_class', result['DAT_class'])
        result['NET_class'] = corrected.get('NET_class', result['NET_class'])
        result['SERT_class'] = corrected.get('SERT_class', result['SERT_class'])

        # Update hERG if corrected
        if corrected.get('herg_corrected'):
            result['herg_risk'] = corrected.get('herg_risk', result.get('herg_risk'))

        # Store drug class
        result['drug_class'] = corrected.get('drug_class')

        # Abuse score - use dedicated predictor with corrected MAT values
        from abuse_predictor import AbusePredictor
        abuse_predictor = AbusePredictor()

        dat_probs = result['DAT_probs']
        net_probs = result['NET_probs']
        sert_probs = result['SERT_probs']
        dat_class = result['DAT_class']  # Now using corrected value
        net_class = result['NET_class']
        sert_class = result['SERT_class']

        score, category = abuse_predictor.predict(
            smiles=smiles,
            dat_class=dat_class,
            net_class=net_class,
            sert_class=sert_class,
            dat_probs=dat_probs,
            net_probs=net_probs,
            sert_probs=sert_probs,
        )
        result['abuse_score'] = score
        result['abuse_category'] = category

    return result


def validate_prediction(pred: dict, expected: dict) -> dict:
    """Compare prediction to expected values."""
    issues = []
    matches = []

    # DAT
    if 'DAT_class' in pred:
        if pred['DAT_class'] == expected['expected_dat']:
            matches.append(f"DAT: {pred['DAT_class']} [OK]")
        else:
            issues.append(f"DAT: predicted {pred['DAT_class']}, expected {expected['expected_dat']}")

    # NET
    if 'NET_class' in pred:
        if pred['NET_class'] == expected['expected_net']:
            matches.append(f"NET: {pred['NET_class']} [OK]")
        else:
            issues.append(f"NET: predicted {pred['NET_class']}, expected {expected['expected_net']}")

    # SERT
    if 'SERT_class' in pred:
        if pred['SERT_class'] == expected['expected_sert']:
            matches.append(f"SERT: {pred['SERT_class']} [OK]")
        else:
            issues.append(f"SERT: predicted {pred['SERT_class']}, expected {expected['expected_sert']}")

    # Abuse
    if 'abuse_category' in pred:
        if pred['abuse_category'] == expected['expected_abuse']:
            matches.append(f"Abuse: {pred['abuse_category']} [OK]")
        else:
            issues.append(f"Abuse: predicted {pred['abuse_category']}, expected {expected['expected_abuse']}")

    # hERG
    if 'herg_risk' in pred:
        if pred['herg_risk'] == expected['expected_herg']:
            matches.append(f"hERG: {pred['herg_risk']} [OK]")
        else:
            issues.append(f"hERG: predicted {pred['herg_risk']}, expected {expected['expected_herg']}")

    return {
        'matches': matches,
        'issues': issues,
        'score': len(matches) / (len(matches) + len(issues)) if (matches or issues) else 0
    }


def main():
    print("=" * 70)
    print("STIMULANT & ADHD MEDICATION VALIDATION SUITE")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")
    models = load_models()
    print()

    # Results storage
    all_results = []

    # Process each drug
    print("Testing compounds...")
    print("-" * 70)

    for name, info in STIMULANT_PHARMACOLOGY.items():
        # Get SMILES
        smiles = info['smiles']
        if smiles is None:
            smiles = fetch_smiles(name)
            if smiles is None:
                print(f"[SKIP] {name}: Could not fetch SMILES")
                continue

        # Predict
        pred = predict_compound(smiles, models)

        if 'error' in pred:
            print(f"[ERROR] {name}: {pred['error']}")
            continue

        # Validate
        validation = validate_prediction(pred, info)

        # Display
        status = "[PASS]" if validation['score'] >= 0.8 else ("[PARTIAL]" if validation['score'] >= 0.5 else "[FAIL]")
        print(f"\n{name.upper()} ({info['class']}) - {status}")
        print(f"  SMILES: {smiles[:50]}...")
        print(f"  Notes: {info['notes']}")

        print(f"  Predictions:")
        print(f"    DAT: {pred.get('DAT_class', 'N/A')} (expected: {info['expected_dat']})")
        print(f"    NET: {pred.get('NET_class', 'N/A')} (expected: {info['expected_net']})")
        print(f"    SERT: {pred.get('SERT_class', 'N/A')} (expected: {info['expected_sert']})")
        print(f"    Abuse: {pred.get('abuse_category', 'N/A')} [{pred.get('abuse_score', 0):.0f}/100] (expected: {info['expected_abuse']})")
        print(f"    hERG: {pred.get('herg_risk', 'N/A')} [{pred.get('herg_prob', 0):.2f}] (expected: {info['expected_herg']})")

        if validation['issues']:
            print(f"  Issues: {', '.join(validation['issues'])}")

        # Store result
        all_results.append({
            'name': name,
            'class': info['class'],
            'smiles': smiles,
            'DAT_pred': pred.get('DAT_class'),
            'DAT_exp': info['expected_dat'],
            'NET_pred': pred.get('NET_class'),
            'NET_exp': info['expected_net'],
            'SERT_pred': pred.get('SERT_class'),
            'SERT_exp': info['expected_sert'],
            'abuse_pred': pred.get('abuse_category'),
            'abuse_exp': info['expected_abuse'],
            'abuse_score': pred.get('abuse_score'),
            'herg_pred': pred.get('herg_risk'),
            'herg_exp': info['expected_herg'],
            'herg_prob': pred.get('herg_prob'),
            'validation_score': validation['score'],
            'issues': '; '.join(validation['issues']),
        })

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(all_results)

    # Overall accuracy
    print(f"\nOverall validation score: {df['validation_score'].mean():.1%}")

    # Per-target accuracy
    for col in ['DAT', 'NET', 'SERT', 'abuse', 'herg']:
        pred_col = f'{col}_pred'
        exp_col = f'{col}_exp'
        if pred_col in df.columns and exp_col in df.columns:
            acc = (df[pred_col] == df[exp_col]).mean()
            print(f"  {col.upper()}: {acc:.1%} accuracy")

    # Problematic predictions
    print("\n" + "-" * 70)
    print("COMPOUNDS WITH ISSUES:")
    issues_df = df[df['validation_score'] < 0.8]
    if len(issues_df) > 0:
        for _, row in issues_df.iterrows():
            print(f"  - {row['name']}: {row['issues']}")
    else:
        print("  None - all predictions match expected pharmacology!")

    # Save results
    output_path = Path(__file__).parent / 'validation_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df


if __name__ == "__main__":
    main()
