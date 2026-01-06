"""
External Validation of Abuse Liability Predictor
=================================================
Tests on extensive compound sets from DEA schedules, DrugBank, and literature.

This is TRUE external validation - compounds NOT used in development.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))

# Extensive compound database with known abuse potential
# Sources: DEA scheduling, DrugBank, clinical literature, FDA labels

EXTERNAL_VALIDATION_SET = {
    # ============================================
    # HIGH ABUSE POTENTIAL (DEA Schedule I-II)
    # ============================================

    # Amphetamines & derivatives
    "phentermine": {"smiles": "CC(N)Cc1ccccc1", "abuse": "HIGH", "class": "Amphetamine", "source": "DEA Schedule IV"},
    "benzphetamine": {"smiles": "CC(Cc1ccccc1)NCc2ccccc2", "abuse": "HIGH", "class": "Amphetamine", "source": "DEA Schedule III"},
    "phenmetrazine": {"smiles": "CC1NCCOC1c2ccccc2", "abuse": "HIGH", "class": "Stimulant", "source": "DEA Schedule II"},
    "cathinone": {"smiles": "CC(N)C(=O)c1ccccc1", "abuse": "HIGH", "class": "Cathinone", "source": "DEA Schedule I"},
    "mephedrone": {"smiles": "CNC(C)C(=O)c1ccc(C)cc1", "abuse": "HIGH", "class": "Cathinone", "source": "DEA Schedule I"},
    "methylone": {"smiles": "CNC(C)C(=O)c1ccc2OCOc2c1", "abuse": "HIGH", "class": "Cathinone", "source": "DEA Schedule I"},
    "alpha-pvp": {"smiles": "CCCCC(C(=O)c1ccccc1)N2CCCC2", "abuse": "HIGH", "class": "Cathinone", "source": "DEA Schedule I"},

    # Opioids (HIGH abuse)
    "morphine": {"smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O", "abuse": "HIGH", "class": "Opioid", "source": "DEA Schedule II"},
    "oxycodone": {"smiles": "CN1CCC23C4C(=O)CCC2(C1CC5=C3C(=C(C=C5)O)O4)OC", "abuse": "HIGH", "class": "Opioid", "source": "DEA Schedule II"},
    "hydrocodone": {"smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)OC)OC3C(=O)CC4", "abuse": "HIGH", "class": "Opioid", "source": "DEA Schedule II"},
    "fentanyl": {"smiles": "CCC(=O)N(C1CCN(CC1)CCC2=CC=CC=C2)C3=CC=CC=C3", "abuse": "HIGH", "class": "Opioid", "source": "DEA Schedule II"},
    "heroin": {"smiles": "CC(=O)OC1C=CC2C3CC4=C5C2(C1OC(C)=O)CCN3CC=C4C(=C5)O", "abuse": "HIGH", "class": "Opioid", "source": "DEA Schedule I"},
    "methadone": {"smiles": "CCC(=O)C(CC(C)N(C)C)(C1=CC=CC=C1)C2=CC=CC=C2", "abuse": "HIGH", "class": "Opioid", "source": "DEA Schedule II"},

    # Benzodiazepines (MODERATE-HIGH)
    "alprazolam": {"smiles": "CC1=NN=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4", "abuse": "MODERATE", "class": "Benzodiazepine", "source": "DEA Schedule IV"},
    "diazepam": {"smiles": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3", "abuse": "MODERATE", "class": "Benzodiazepine", "source": "DEA Schedule IV"},
    "lorazepam": {"smiles": "OC1N=C(C2=CC=CC=C2)C3=C(C=C(Cl)C=C3)NC1=O", "abuse": "MODERATE", "class": "Benzodiazepine", "source": "DEA Schedule IV"},
    "clonazepam": {"smiles": "OC1N=C(C2=CC=CC=C2)C3=CC([N+]([O-])=O)=CC=C3NC1=O", "abuse": "MODERATE", "class": "Benzodiazepine", "source": "DEA Schedule IV"},
    "midazolam": {"smiles": "CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F", "abuse": "MODERATE", "class": "Benzodiazepine", "source": "DEA Schedule IV"},

    # Barbiturates
    "phenobarbital": {"smiles": "CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2", "abuse": "MODERATE", "class": "Barbiturate", "source": "DEA Schedule IV"},
    "secobarbital": {"smiles": "CCCC(C)C1(CC=C)C(=O)NC(=O)NC1=O", "abuse": "HIGH", "class": "Barbiturate", "source": "DEA Schedule II"},
    "pentobarbital": {"smiles": "CCCC(C)C1(CC)C(=O)NC(=O)NC1=O", "abuse": "HIGH", "class": "Barbiturate", "source": "DEA Schedule II"},

    # Other stimulants
    "pemoline": {"smiles": "NC1=NC2=C(O1)C(=O)NC2C3=CC=CC=C3", "abuse": "MODERATE", "class": "Stimulant", "source": "DEA Schedule IV"},
    "phendimetrazine": {"smiles": "CC1C(CC(O1)C)NC", "abuse": "MODERATE", "class": "Stimulant", "source": "DEA Schedule III"},

    # Dissociatives
    "ketamine": {"smiles": "CNC1(CCCCC1=O)C2=CC=CC=C2Cl", "abuse": "MODERATE", "class": "Dissociative", "source": "DEA Schedule III"},
    "pcp": {"smiles": "C1CCC(CC1)N2CCCCC2C3=CC=CC=C3", "abuse": "HIGH", "class": "Dissociative", "source": "DEA Schedule II"},

    # Cannabis
    "thc": {"smiles": "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O", "abuse": "MODERATE", "class": "Cannabinoid", "source": "DEA Schedule I"},
    "nabilone": {"smiles": "CCCCCC1=CC(=C2C3CCC(=O)C3C(OC2=C1)(C)C)O", "abuse": "MODERATE", "class": "Cannabinoid", "source": "DEA Schedule II"},

    # ============================================
    # LOW ABUSE POTENTIAL (Non-scheduled / OTC)
    # ============================================

    # Antidepressants (SSRIs) - LOW abuse
    "sertraline": {"smiles": "CNC1CCC(C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl", "abuse": "LOW", "class": "SSRI", "source": "Non-scheduled"},
    "paroxetine": {"smiles": "FC1=CC=C(C=C1)C2CCNCC2COC3=CC4=C(C=C3)OCO4", "abuse": "LOW", "class": "SSRI", "source": "Non-scheduled"},
    "citalopram": {"smiles": "CN(C)CCCC1(C2=CC=C(C=C2)F)OCC3=C1C=CC(=C3)C#N", "abuse": "LOW", "class": "SSRI", "source": "Non-scheduled"},
    "escitalopram": {"smiles": "CN(C)CCC[C@]1(OCC2=C1C=CC(=C2)C#N)C3=CC=C(C=C3)F", "abuse": "LOW", "class": "SSRI", "source": "Non-scheduled"},
    "fluvoxamine": {"smiles": "COCCCCC(=NOCCN)C1=CC=C(C=C1)C(F)(F)F", "abuse": "LOW", "class": "SSRI", "source": "Non-scheduled"},

    # SNRIs - LOW abuse
    "duloxetine": {"smiles": "CNCCC(C1=CC=CS1)OC2=CC=CC3=CC=CC=C32", "abuse": "LOW", "class": "SNRI", "source": "Non-scheduled"},
    "desvenlafaxine": {"smiles": "CN(C)CC(C1=CC=C(C=C1)O)C2(CCCCC2)O", "abuse": "LOW", "class": "SNRI", "source": "Non-scheduled"},
    "levomilnacipran": {"smiles": "CCN[C@@H]1C[C@H]1(C(=O)N)C2=CC=CC=C2", "abuse": "LOW", "class": "SNRI", "source": "Non-scheduled"},

    # Tricyclics - LOW abuse
    "amitriptyline": {"smiles": "CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31", "abuse": "LOW", "class": "TCA", "source": "Non-scheduled"},
    "nortriptyline": {"smiles": "CNCCC=C1C2=CC=CC=C2CCC3=CC=CC=C31", "abuse": "LOW", "class": "TCA", "source": "Non-scheduled"},
    "imipramine": {"smiles": "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31", "abuse": "LOW", "class": "TCA", "source": "Non-scheduled"},
    "desipramine": {"smiles": "CNCCCN1C2=CC=CC=C2CCC3=CC=CC=C31", "abuse": "LOW", "class": "TCA", "source": "Non-scheduled"},
    "clomipramine": {"smiles": "CN(C)CCCN1C2=CC=CC=C2CC3=C1C=C(C=C3)Cl", "abuse": "LOW", "class": "TCA", "source": "Non-scheduled"},

    # Antipsychotics - LOW abuse
    "olanzapine": {"smiles": "CC1=CC2=C(S1)NC3=C(N2CC4=CN=CN4)C=C(C=C3)N5CCCCC5", "abuse": "LOW", "class": "Antipsychotic", "source": "Non-scheduled"},
    "risperidone": {"smiles": "CC1=C(C=CC(=C1)C2=NOC3=C2CCC(=O)N3)N4CCC(CC4)N5C(=O)NC6=CC=CC=C65", "abuse": "LOW", "class": "Antipsychotic", "source": "Non-scheduled"},
    "quetiapine": {"smiles": "OCCOCCN1CCN(CC1)C2=NC3=CC=CC=C3SC4=CC=CC=C42", "abuse": "LOW", "class": "Antipsychotic", "source": "Non-scheduled"},
    "aripiprazole": {"smiles": "ClC1=CC=CC(=C1)N2CCN(CC2)CCCCOC3=CC=C4CCC(=O)NC4=C3", "abuse": "LOW", "class": "Antipsychotic", "source": "Non-scheduled"},
    "ziprasidone": {"smiles": "ClC1=CC=C2C(=C1)SC3=C2CCN(C3)CCCN4C(=O)C5=CC=CC=C5N=C4", "abuse": "LOW", "class": "Antipsychotic", "source": "Non-scheduled"},
    "chlorpromazine": {"smiles": "CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl", "abuse": "LOW", "class": "Antipsychotic", "source": "Non-scheduled"},

    # Anticonvulsants - LOW abuse
    "carbamazepine": {"smiles": "NC(=O)N1C2=CC=CC=C2C=CC3=CC=CC=C31", "abuse": "LOW", "class": "Anticonvulsant", "source": "Non-scheduled"},
    "lamotrigine": {"smiles": "NC1=NC(=C(N=N1)C2=CC=CC(=C2)Cl)C3=CC=CC=C3Cl", "abuse": "LOW", "class": "Anticonvulsant", "source": "Non-scheduled"},
    "topiramate": {"smiles": "CC1(C)OC2COC3(COS(N)(=O)=O)OC(C)(C)OC3C2O1", "abuse": "LOW", "class": "Anticonvulsant", "source": "Non-scheduled"},
    "valproic_acid": {"smiles": "CCCC(CCC)C(=O)O", "abuse": "LOW", "class": "Anticonvulsant", "source": "Non-scheduled"},
    "levetiracetam": {"smiles": "CC[C@@H](C(=O)N)N1CCCC1=O", "abuse": "LOW", "class": "Anticonvulsant", "source": "Non-scheduled"},

    # Muscle relaxants - LOW/MODERATE
    "cyclobenzaprine": {"smiles": "CN(C)CCC=C1C2=CC=CC=C2C=CC3=CC=CC=C31", "abuse": "LOW", "class": "Muscle Relaxant", "source": "Non-scheduled"},
    "baclofen": {"smiles": "NCC(CC1=CC=C(C=C1)Cl)C(O)=O", "abuse": "LOW", "class": "Muscle Relaxant", "source": "Non-scheduled"},
    "carisoprodol": {"smiles": "CCCN(CCC)C(=O)OCC(CCC)(COC(=O)N)C", "abuse": "MODERATE", "class": "Muscle Relaxant", "source": "DEA Schedule IV"},

    # Antihistamines - LOW
    "diphenhydramine": {"smiles": "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2", "abuse": "LOW", "class": "Antihistamine", "source": "OTC"},
    "hydroxyzine": {"smiles": "OCCOCCN1CCN(CC1)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl", "abuse": "LOW", "class": "Antihistamine", "source": "Non-scheduled"},
    "promethazine": {"smiles": "CC(CN1C2=CC=CC=C2SC3=CC=CC=C31)N(C)C", "abuse": "LOW", "class": "Antihistamine", "source": "Non-scheduled"},
    "cetirizine": {"smiles": "OC(=O)COCCN1CCN(CC1)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl", "abuse": "LOW", "class": "Antihistamine", "source": "OTC"},

    # Beta blockers - LOW
    "propranolol": {"smiles": "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C21", "abuse": "LOW", "class": "Beta Blocker", "source": "Non-scheduled"},
    "metoprolol": {"smiles": "CC(C)NCC(O)COC1=CC=C(C=C1)CCOC", "abuse": "LOW", "class": "Beta Blocker", "source": "Non-scheduled"},
    "atenolol": {"smiles": "CC(C)NCC(O)COC1=CC=C(C=C1)CC(N)=O", "abuse": "LOW", "class": "Beta Blocker", "source": "Non-scheduled"},

    # ACE inhibitors - LOW
    "lisinopril": {"smiles": "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", "abuse": "LOW", "class": "ACE Inhibitor", "source": "Non-scheduled"},
    "enalapril": {"smiles": "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(=O)O", "abuse": "LOW", "class": "ACE Inhibitor", "source": "Non-scheduled"},

    # Statins - LOW
    "atorvastatin": {"smiles": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", "abuse": "LOW", "class": "Statin", "source": "Non-scheduled"},
    "simvastatin": {"smiles": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21", "abuse": "LOW", "class": "Statin", "source": "Non-scheduled"},

    # NSAIDs - LOW
    "ibuprofen": {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "abuse": "LOW", "class": "NSAID", "source": "OTC"},
    "naproxen": {"smiles": "COC1=CC2=CC(=CC=C2C=C1)C(C)C(=O)O", "abuse": "LOW", "class": "NSAID", "source": "OTC"},
    "celecoxib": {"smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", "abuse": "LOW", "class": "NSAID", "source": "Non-scheduled"},

    # Antibiotics - LOW
    "amoxicillin": {"smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O", "abuse": "LOW", "class": "Antibiotic", "source": "Non-scheduled"},
    "azithromycin": {"smiles": "CC1CC(C(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C", "abuse": "LOW", "class": "Antibiotic", "source": "Non-scheduled"},

    # Proton pump inhibitors - LOW
    "omeprazole": {"smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC", "abuse": "LOW", "class": "PPI", "source": "OTC"},
    "pantoprazole": {"smiles": "COC1=CC=NC2=C1C(=CN2S(=O)CC3=C(C(=CC=C3)OC)OC)C", "abuse": "LOW", "class": "PPI", "source": "Non-scheduled"},

    # Anti-diabetics - LOW
    "metformin": {"smiles": "CN(C)C(=N)NC(=N)N", "abuse": "LOW", "class": "Antidiabetic", "source": "Non-scheduled"},
    "glipizide": {"smiles": "CC1=NC=C(C=C1)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3", "abuse": "LOW", "class": "Antidiabetic", "source": "Non-scheduled"},

    # ============================================
    # MODERATE ABUSE POTENTIAL
    # ============================================

    # Z-drugs (sleep aids)
    "zolpidem": {"smiles": "CC1=CC=C(C=C1)C2=C(N3C=C(C)C(C)=CN3C(=O)C2)C4=CC=CC=C4", "abuse": "MODERATE", "class": "Z-drug", "source": "DEA Schedule IV"},
    "zaleplon": {"smiles": "CCN(C1=CC=CC(=C1)C2=CC=NC3=C(C=NN23)C#N)C(=O)C", "abuse": "MODERATE", "class": "Z-drug", "source": "DEA Schedule IV"},
    "eszopiclone": {"smiles": "CC(=O)OC1C(OC(CN2C=NC3=C2C(=O)N(C=N3)C4=CC(=CC=C4)Cl)C1O)CO", "abuse": "MODERATE", "class": "Z-drug", "source": "DEA Schedule IV"},

    # Gabapentinoids
    "pregabalin": {"smiles": "CC(C)CC(CC(=O)O)CN", "abuse": "MODERATE", "class": "Gabapentinoid", "source": "DEA Schedule V"},
    "gabapentin": {"smiles": "NCC1(CCCCC1)CC(=O)O", "abuse": "LOW", "class": "Gabapentinoid", "source": "Non-scheduled"},

    # Tramadol
    "tramadol": {"smiles": "CN(C)C[C@H]1CCCC[C@]1(C2=CC(=CC=C2)OC)O", "abuse": "MODERATE", "class": "Opioid", "source": "DEA Schedule IV"},
}


def fetch_smiles_pubchem(name: str) -> str:
    """Fetch SMILES from PubChem if not available."""
    try:
        import urllib.parse
        encoded = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data['PropertyTable']['Properties'][0].get('IsomericSMILES')
    except:
        pass
    return None


def load_models():
    """Load prediction models."""
    from model import StereoGNN
    from featurizer import MoleculeGraphFeaturizer

    models = {}
    model_dir = Path(__file__).parent / 'models'

    # MAT model
    mat_path = Path(__file__).parent / 'best_model.pt'
    if mat_path.exists():
        import torch
        from app import StereoGNNSmallFinetune
        models['mat'] = StereoGNNSmallFinetune()
        ckpt = torch.load(mat_path, map_location='cpu', weights_only=False)
        models['mat'].load_state_dict(ckpt['model_state_dict'])
        models['mat'].eval()
        print("  [OK] MAT model loaded")

    models['featurizer'] = MoleculeGraphFeaturizer(use_3d=False)
    return models


def predict_abuse(smiles: str, models: dict) -> Tuple[str, float]:
    """Predict abuse potential for a compound."""
    import torch
    from torch_geometric.data import Batch
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    import numpy as np

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Featurize
    data = models['featurizer'].featurize(smiles, {'DAT': -1, 'NET': -1, 'SERT': -1})
    if data is None:
        return None, None

    # Fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_arr = np.zeros(1024)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    desc = [
        Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.MolWt(mol) / 500,
        Descriptors.NumHDonors(mol) / 5, Descriptors.NumHAcceptors(mol) / 10,
        Descriptors.NumRotatableBonds(mol) / 10, Descriptors.NumAromaticRings(mol) / 5,
        Descriptors.FractionCSP3(mol),
    ]
    data.fp = torch.tensor(np.concatenate([fp_arr, desc]).astype(np.float32)).unsqueeze(0)

    batch = Batch.from_data_list([data])

    # Get MAT predictions
    with torch.no_grad():
        mat_output = models['mat'](batch)

        result = {}
        for target in ['DAT', 'NET', 'SERT']:
            logits = mat_output[target][0]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax().item()
            result[f'{target}_class'] = ['inactive', 'blocker', 'substrate'][pred_idx]
            result[f'{target}_probs'] = probs.numpy()

    # Apply pharmacology rules
    from pharmacology_rules import PharmacologyRules
    rules = PharmacologyRules()
    corrected = rules.correct_predictions(smiles, result)

    for target in ['DAT', 'NET', 'SERT']:
        result[f'{target}_class'] = corrected.get(f'{target}_class', result[f'{target}_class'])

    # Use abuse predictor
    from abuse_predictor import AbusePredictor
    abuse_pred = AbusePredictor()

    score, category = abuse_pred.predict(
        smiles=smiles,
        dat_class=result['DAT_class'],
        net_class=result['NET_class'],
        sert_class=result['SERT_class'],
        dat_probs=result['DAT_probs'],
        net_probs=result['NET_probs'],
        sert_probs=result['SERT_probs'],
    )

    return category, score


def main():
    print("=" * 70)
    print("EXTERNAL VALIDATION: Abuse Liability Predictor")
    print("=" * 70)
    print(f"\nValidation set: {len(EXTERNAL_VALIDATION_SET)} compounds")
    print("Sources: DEA Schedules, DrugBank, FDA labels, Clinical literature\n")

    # Load models
    print("Loading models...")
    models = load_models()
    print()

    # Run predictions
    results = []
    correct = 0
    total = 0

    print("Running predictions...")
    print("-" * 70)

    for name, info in EXTERNAL_VALIDATION_SET.items():
        smiles = info['smiles']
        expected = info['abuse']
        drug_class = info['class']

        if not smiles:
            smiles = fetch_smiles_pubchem(name)
            if not smiles:
                print(f"  [SKIP] {name}: Could not get SMILES")
                continue

        predicted, score = predict_abuse(smiles, models)

        if predicted is None:
            print(f"  [SKIP] {name}: Prediction failed")
            continue

        match = predicted == expected
        total += 1
        if match:
            correct += 1

        status = "[OK]" if match else "[MISS]"
        print(f"  {status} {name}: {predicted} (expected: {expected}) - {drug_class}")

        results.append({
            'name': name,
            'class': drug_class,
            'source': info['source'],
            'expected': expected,
            'predicted': predicted,
            'score': score,
            'correct': match,
        })

    # Calculate metrics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.1%}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for abuse_level in ['HIGH', 'MODERATE', 'LOW']:
        subset = df[df['expected'] == abuse_level]
        if len(subset) > 0:
            class_acc = subset['correct'].mean()
            print(f"  {abuse_level}: {class_acc:.1%} ({subset['correct'].sum()}/{len(subset)})")

    # Calculate AUC (binary: HIGH vs LOW)
    print("\nAUC Analysis:")

    # HIGH vs not HIGH
    df['is_high'] = (df['expected'] == 'HIGH').astype(int)
    df['pred_high'] = (df['predicted'] == 'HIGH').astype(int)
    df['score_normalized'] = df['score'] / 100

    try:
        high_auc = roc_auc_score(df['is_high'], df['score_normalized'])
        print(f"  HIGH vs Others AUC: {high_auc:.4f}")
    except:
        print("  HIGH vs Others AUC: N/A")

    # LOW vs not LOW
    df['is_low'] = (df['expected'] == 'LOW').astype(int)
    df['pred_low'] = (df['predicted'] == 'LOW').astype(int)
    df['score_inv'] = 1 - df['score_normalized']

    try:
        low_auc = roc_auc_score(df['is_low'], df['score_inv'])
        print(f"  LOW vs Others AUC: {low_auc:.4f}")
    except:
        print("  LOW vs Others AUC: N/A")

    # Confusion matrix
    print("\nConfusion Matrix:")
    labels = ['HIGH', 'MODERATE', 'LOW']
    cm = confusion_matrix(df['expected'], df['predicted'], labels=labels)
    print(f"             Predicted")
    print(f"             HIGH  MOD   LOW")
    for i, label in enumerate(labels):
        print(f"  Actual {label:4s}  {cm[i][0]:3d}   {cm[i][1]:3d}   {cm[i][2]:3d}")

    # By drug class
    print("\nAccuracy by drug class:")
    for drug_class in df['class'].unique():
        subset = df[df['class'] == drug_class]
        if len(subset) >= 2:
            class_acc = subset['correct'].mean()
            print(f"  {drug_class}: {class_acc:.1%} ({len(subset)} compounds)")

    # Save results
    output_path = Path(__file__).parent / 'external_validation_abuse_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"""
Validation on {total} external compounds:
- Accuracy: {accuracy:.1%}
- This is TRUE external validation (compounds not used in development)

Target performance:
- >85% accuracy indicates strong generalization
- >0.85 AUC indicates reliable ranking of abuse potential
""")

    return df


if __name__ == "__main__":
    main()
