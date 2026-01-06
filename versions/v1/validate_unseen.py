"""
Validation on Unseen Drugs
==========================
Tests model generalization on drugs unlikely to be in training data.

These are newer drugs, unique scaffolds, or drugs from different therapeutic areas
that probably weren't in the ChEMBL transporter dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import requests
import urllib.parse

sys.path.insert(0, str(Path(__file__).parent))

from app import StereoGNNSmallFinetune
from featurizer import MoleculeGraphFeaturizer
from torch_geometric.data import Batch

# Drugs unlikely to be in transporter training data
# with known pharmacology from literature
UNSEEN_DRUGS = {
    # ===== NEWER ADHD MEDICATIONS =====
    "viloxazine": {
        "expected_dat": "inactive",
        "expected_net": "blocker",  # Selective NRI
        "expected_sert": "inactive",
        "notes": "Qelbree (2021) - selective NRI for ADHD, not a stimulant",
        "source": "FDA label, Nasser 2021"
    },
    "solriamfetol": {
        "expected_dat": "blocker",  # Weak DAT/NET inhibitor
        "expected_net": "blocker",
        "expected_sert": "inactive",
        "notes": "Sunosi (2019) - DNRI for narcolepsy, not a releaser",
        "source": "FDA label"
    },

    # ===== ATYPICAL ANTIDEPRESSANTS =====
    "vortioxetine": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "blocker",  # Multimodal - SERT inhibitor + 5-HT modulator
        "notes": "Trintellix - multimodal antidepressant",
        "source": "Sanchez 2015"
    },
    "vilazodone": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "blocker",  # SSRI + 5-HT1A partial agonist
        "notes": "Viibryd - SSRI/5-HT1A agonist",
        "source": "FDA label"
    },
    "levomilnacipran": {
        "expected_dat": "inactive",
        "expected_net": "blocker",  # More NET than SERT selective
        "expected_sert": "blocker",
        "notes": "Fetzima - SNRI with NET preference",
        "source": "Auclair 2013"
    },
    "desvenlafaxine": {
        "expected_dat": "inactive",
        "expected_net": "blocker",
        "expected_sert": "blocker",
        "notes": "Pristiq - active metabolite of venlafaxine",
        "source": "FDA label"
    },

    # ===== NOVEL MECHANISM DRUGS =====
    "pitolisant": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "notes": "Wakix - H3 receptor antagonist, no MAT activity",
        "source": "FDA label"
    },
    "lemborexant": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "notes": "Dayvigo - orexin receptor antagonist",
        "source": "FDA label"
    },
    "esketamine": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",  # NMDA antagonist, minimal MAT
        "notes": "Spravato - NMDA antagonist, S-enantiomer of ketamine",
        "source": "Zanos 2018"
    },

    # ===== DRUGS OF ABUSE (newer synthetic) =====
    "mephedrone": {
        "expected_dat": "substrate",  # Synthetic cathinone
        "expected_net": "substrate",
        "expected_sert": "substrate",
        "notes": "4-MMC - synthetic cathinone, triple releaser",
        "source": "Baumann 2012"
    },
    "alpha-pvp": {
        "expected_dat": "blocker",  # Pyrrolidine cathinone - blocker not releaser
        "expected_net": "blocker",
        "expected_sert": "inactive",
        "notes": "Flakka - pyrovalerone analog, DAT/NET blocker",
        "source": "Marusich 2014"
    },

    # ===== NEGATIVE CONTROLS =====
    "pregabalin": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "notes": "Lyrica - calcium channel modulator, no MAT activity",
        "source": "FDA label"
    },
    "gabapentin": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "notes": "Neurontin - GABA analog, no MAT activity",
        "source": "FDA label"
    },
    "lamotrigine": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "notes": "Lamictal - sodium channel blocker",
        "source": "FDA label"
    },
    "topiramate": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",
        "notes": "Topamax - multiple mechanisms, no MAT",
        "source": "FDA label"
    },

    # ===== APPETITE SUPPRESSANTS =====
    "lorcaserin": {
        "expected_dat": "inactive",
        "expected_net": "inactive",
        "expected_sert": "inactive",  # 5-HT2C agonist, not SERT
        "notes": "Belviq (withdrawn) - 5-HT2C agonist",
        "source": "FDA label"
    },
    "phentermine": {
        "expected_dat": "substrate",  # Amphetamine-like
        "expected_net": "substrate",
        "expected_sert": "inactive",
        "notes": "Adipex - amphetamine congener for weight loss",
        "source": "Rothman 2001"
    },
}


def fetch_smiles(name: str) -> str:
    """Fetch SMILES from PubChem."""
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


def main():
    print("=" * 70)
    print("VALIDATION ON UNSEEN DRUGS")
    print("=" * 70)
    print("Testing generalization to drugs not in training data\n")

    # Load model
    print("Loading model...")
    model = StereoGNNSmallFinetune()
    ckpt = torch.load(Path(__file__).parent / 'best_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model AUC (validation): {ckpt.get('best_auc', 0):.3f}\n")

    featurizer = MoleculeGraphFeaturizer(use_3d=False)

    results = []
    print("-" * 70)

    for drug_name, expected in UNSEEN_DRUGS.items():
        # Fetch SMILES
        smiles = fetch_smiles(drug_name)
        if not smiles:
            print(f"[SKIP] {drug_name}: Could not fetch SMILES")
            continue

        # Featurize
        try:
            data = featurizer.featurize(smiles, {'DAT': -1, 'NET': -1, 'SERT': -1})
            if data is None:
                print(f"[SKIP] {drug_name}: Featurization failed")
                continue
            batch = Batch.from_data_list([data])
        except Exception as e:
            print(f"[SKIP] {drug_name}: {e}")
            continue

        # Predict
        with torch.no_grad():
            output = model(batch)

        # Get raw predictions
        raw_preds = {}
        for target in ['DAT', 'NET', 'SERT']:
            probs = torch.softmax(output[target], dim=-1)[0]
            pred_class = ['inactive', 'blocker', 'substrate'][probs.argmax().item()]
            raw_preds[target] = {
                'class': pred_class,
                'probs': probs.numpy(),
                'max_prob': probs.max().item(),
            }

        # Apply pharmacology rules
        from app_ui import apply_pharmacology_rules
        corrected = apply_pharmacology_rules(smiles, raw_preds)

        # Analyze
        correct = 0
        total = 0
        issues = []

        drug_class = corrected.get('drug_class', 'unknown')
        print(f"\n{drug_name.upper()} [detected: {drug_class}]")
        print(f"  Notes: {expected['notes']}")
        print(f"  Source: {expected['source']}")

        for target in ['DAT', 'NET', 'SERT']:
            pred_class = corrected[target]['class']
            exp_class = expected[f'expected_{target.lower()}']
            confidence = corrected[target]['max_prob']
            was_corrected = corrected[target].get('corrected', False)

            match = pred_class == exp_class
            if match:
                correct += 1
            else:
                issues.append(f"{target}: {pred_class} vs {exp_class}")
            total += 1

            conf_indicator = "RULE" if was_corrected else ("HIGH" if confidence > 0.8 else ("MED" if confidence > 0.6 else "LOW"))
            status = "OK" if match else "WRONG"
            print(f"  {target}: {pred_class:10} (exp: {exp_class:10}) [{conf_indicator}] {status}")

        accuracy = correct / total
        results.append({
            'drug': drug_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'issues': '; '.join(issues),
            'notes': expected['notes'],
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No drugs could be tested!")
        return

    overall_correct = df['correct'].sum()
    overall_total = df['total'].sum()
    overall_acc = overall_correct / overall_total

    print(f"\nOverall accuracy: {overall_correct}/{overall_total} = {overall_acc:.1%}")

    # Perfect predictions
    perfect = df[df['accuracy'] == 1.0]
    print(f"\nPerfect predictions ({len(perfect)}/{len(df)}):")
    for _, row in perfect.iterrows():
        print(f"  {row['drug']}")

    # Problem predictions
    problems = df[df['accuracy'] < 1.0]
    print(f"\nPredictions with issues ({len(problems)}/{len(df)}):")
    for _, row in problems.iterrows():
        print(f"  {row['drug']}: {row['issues']}")

    # Save
    output_path = Path(__file__).parent / 'validation_unseen_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
