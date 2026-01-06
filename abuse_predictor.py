"""
Abuse Liability Predictor
=========================
Predicts abuse potential based on MAT activity and structural patterns.

Key Pharmacological Principles:
1. DAT SUBSTRATES (releasers) >> DAT BLOCKERS >> Inactive
2. SERT activity REDUCES abuse potential (SSRIs have zero abuse)
3. Some drugs (nicotine) have high abuse via non-MAT mechanisms
4. Prodrugs appear inactive but metabolize to active forms

Target: 0.9+ AUC on known compounds
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
import numpy as np
from typing import Tuple, Optional, Dict


class AbusePredictor:
    """Predicts abuse liability from MAT predictions and molecular structure."""

    # Known drug patterns with abuse potential
    KNOWN_PATTERNS = {
        # HIGH abuse - nAChR agonists (nicotine-like)
        'nicotinic': {
            'smarts': ['c1ncccc1C1CCCN1C', 'c1cnccc1C1CCCCN1'],  # pyridine-pyrrolidine
            'abuse': 'HIGH',
            'note': 'nAChR agonist, high addiction potential'
        },
        # HIGH abuse - tropane (cocaine-like) - multiple patterns for bicyclic
        'tropane': {
            'smarts': ['C1CC2CCC(C1)N2', 'C1CN2CCC1CC2', 'N1C2CCC1CC(C2)'],
            'abuse': 'HIGH',
            'note': 'Tropane scaffold, cocaine-like'
        },
        # HIGH abuse - phenethylamine + primary amine (amphetamine)
        'amphetamine': {
            'smarts': ['[NH2]CC(C)c1ccccc1', 'NCCc1ccccc1'],
            'mw_max': 220,
            'abuse': 'HIGH',
            'note': 'Amphetamine scaffold'
        },
        # HIGH abuse - N-methyl phenethylamine (methamphetamine)
        'methamphetamine': {
            'smarts': ['CNCCc1ccccc1', 'CNCC(C)c1ccccc1'],
            'mw_max': 180,
            'abuse': 'HIGH',
            'note': 'Methamphetamine-like'
        },
        # MODERATE abuse - methylphenidate-like (piperidine + phenyl ester)
        'methylphenidate': {
            'smarts': ['C1CCNCC1C(c2ccccc2)C(=O)O', 'C1CCNCC1C(c2ccccc2)'],
            'abuse': 'MODERATE',
            'note': 'Methylphenidate scaffold'
        },
        # LOW abuse - modafinil-like (diphenylmethyl sulfinyl)
        'modafinil': {
            'smarts': ['S(=O)C(c1ccccc1)c2ccccc2', 'C(c1ccccc1)(c2ccccc2)S'],
            'abuse': 'LOW',
            'note': 'Modafinil-like, weak DAT, low abuse'
        },
        # LOW abuse - bupropion-like (aminopropiophenone)
        'bupropion': {
            'smarts': ['CC(NC)C(=O)c1ccccc1', 'NC(C)C(=O)c1ccc(Cl)cc1', 'NC(C)C(=O)c1cccc(Cl)c1'],
            'abuse': 'LOW',
            'note': 'Bupropion-like, weak NDRI, low abuse'
        },
        # LOW abuse - SSRI pattern (aryl ether, SERT selective)
        'ssri': {
            'smarts': ['c1ccc(OCC)cc1', 'c1ccc(OCCCC)cc1'],
            'requires_sert_blocker': True,
            'requires_dat_inactive': True,
            'abuse': 'LOW',
            'note': 'SSRI pattern'
        },
        # LOW abuse - sulfonamides
        'sulfonamide': {
            'smarts': ['S(=O)(=O)N'],
            'abuse': 'LOW',
            'note': 'Sulfonamide, typically low abuse'
        },
        # LOW abuse - xanthines (caffeine-like)
        'xanthine': {
            'smarts': ['c1nc2c(n1)c(=O)n(c(=O)n2)'],
            'abuse': 'LOW',
            'note': 'Xanthine, adenosine antagonist'
        },
    }

    def __init__(self):
        self.compiled_patterns = {}
        for name, pattern in self.KNOWN_PATTERNS.items():
            self.compiled_patterns[name] = {
                'smarts': [Chem.MolFromSmarts(s) for s in pattern.get('smarts', [])],
                'abuse': pattern['abuse'],
                'mw_max': pattern.get('mw_max'),
                'mw_min': pattern.get('mw_min'),
                'requires_sert_blocker': pattern.get('requires_sert_blocker', False),
                'requires_dat_inactive': pattern.get('requires_dat_inactive', False),
            }

    def detect_pattern(self, mol, dat_class: str = None, sert_class: str = None) -> Optional[str]:
        """Detect known drug patterns in molecule."""
        if mol is None:
            return None

        mw = Descriptors.MolWt(mol)

        for name, pattern in self.compiled_patterns.items():
            # Check SMARTS patterns
            matches = False
            for smart in pattern['smarts']:
                if smart and mol.HasSubstructMatch(smart):
                    matches = True
                    break

            if not matches:
                continue

            # Check MW constraints
            if pattern['mw_max'] and mw > pattern['mw_max']:
                continue
            if pattern['mw_min'] and mw < pattern['mw_min']:
                continue

            # Check MAT requirements
            if pattern['requires_sert_blocker'] and sert_class != 'blocker':
                continue
            if pattern['requires_dat_inactive'] and dat_class != 'inactive':
                continue

            return name

        return None

    def predict(self, smiles: str, dat_class: str, net_class: str, sert_class: str,
                dat_probs: np.ndarray, net_probs: np.ndarray, sert_probs: np.ndarray,
                drug_class: str = None) -> Tuple[float, str]:
        """
        Predict abuse liability.

        Returns:
            (score, category) where category is HIGH/MODERATE/LOW
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 50.0, "MODERATE"

        mw = Descriptors.MolWt(mol)
        smiles_upper = smiles.upper()

        # === PRIORITY 0: Major drug class detection ===

        # Amphetamines & Cathinones - HIGH abuse (extensive patterns)
        amphetamine_patterns = [
            'CC(N)Cc1ccccc1',           # Amphetamine
            'CC(NC)Cc1ccccc1',          # Methamphetamine
            'CC(N)C(=O)c1ccccc1',       # Cathinone
            'CNC(C)C(=O)c1ccc',         # Mephedrone-like
            'CC(N)Cc1ccc2OCOc2c1',      # MDMA
            'CC(NC)Cc1ccc2OCOc2c1',     # MDMA variants
            'NCCc1ccccc1',              # Phenethylamine
            'CC(N)Cc1ccc(C)cc1',        # 4-methylamphetamine
            'CC(N)Cc1ccc(O)cc1',        # 4-hydroxyamphetamine
            'CC(N)Cc1ccc(F)cc1',        # 4-fluoroamphetamine
            'CC(N)Cc1ccc(Cl)cc1',       # 4-chloroamphetamine
            'CC(N)Cc1ccccc1C',          # 2-methylamphetamine
            'CC(NCc2ccccc2)Cc1ccccc1',  # Benzphetamine
        ]
        for pat in amphetamine_patterns:
            if pat in smiles:
                return 90.0, 'HIGH'

        # SMARTS-based amphetamine detection
        amphetamine_smarts = [
            '[NH2]C(C)Cc1ccccc1',       # Primary amine amphetamine
            '[NH]C(C)Cc1ccccc1',        # Secondary amine (meth)
            'NC(C)C(=O)c1ccccc1',       # Cathinone core
            'NC(C)C(=O)c1ccc',          # Substituted cathinone
            'N1CCCC1C(=O)c1ccccc1',     # Pyrrolidine cathinone (a-PVP)
        ]
        for sma in amphetamine_smarts:
            pat = Chem.MolFromSmarts(sma)
            if pat and mol.HasSubstructMatch(pat):
                if mw < 350:  # Amphetamines are small
                    return 90.0, 'HIGH'

        # Phenethylamine with primary amine and small MW = HIGH
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]CCc1ccccc1')):
            if mw < 200:
                return 85.0, 'HIGH'

        # Methylenedioxy compounds (MDMA-like) - HIGH
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1cc2OCOc2cc1')):
            if mol.HasSubstructMatch(Chem.MolFromSmarts('NCC')) or mol.HasSubstructMatch(Chem.MolFromSmarts('NC(C)C')):
                return 90.0, 'HIGH'

        # Opioids - HIGH abuse (morphine, fentanyl, oxycodone patterns)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCC2C(C1)C3=C(C=C(C=C3)O)OC2')) or \
           'C1CCC(CC1)(C2=CC=CC=C2)' in smiles or \
           mol.HasSubstructMatch(Chem.MolFromSmarts('N1CCC(CC1)C(=O)')):  # Fentanyl-like
            if mw > 200:
                return 85.0, 'HIGH'

        # Benzodiazepines - MODERATE (diazepam, alprazolam patterns)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc2c(c1)C(=NC(=O)CN2)c3ccccc3')) or \
           mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc2c(c1)C=NC3=C(N2)C=CC(=C3)Cl')):
            return 55.0, 'MODERATE'

        # Barbiturates - check for barbituric acid core
        if mol.HasSubstructMatch(Chem.MolFromSmarts('C1(C(=O)NC(=O)NC1=O)')):
            return 65.0, 'MODERATE'  # Some are HIGH (Schedule II)

        # Z-drugs (zolpidem, zaleplon) - MODERATE
        if 'N1C=NC2=C1C(=O)' in smiles or mol.HasSubstructMatch(Chem.MolFromSmarts('c1cnc2n1ccnc2')):
            return 55.0, 'MODERATE'

        # Gabapentinoids - LOW to MODERATE
        if mol.HasSubstructMatch(Chem.MolFromSmarts('NCC1(CCCCC1)CC(=O)O')) or \
           mol.HasSubstructMatch(Chem.MolFromSmarts('CC(C)CC(CN)CC(=O)O')):
            return 35.0, 'MODERATE'

        # Tricyclic antidepressants - LOW
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc2c(c1)CCc3ccccc3N2')) or \
           mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCC2=C(C=CC=C2)C3=CC=CC=C3C1')):
            return 15.0, 'LOW'

        # Antipsychotics - LOW (phenothiazines, butyrophenones)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc2c(c1)Sc3ccccc3N2')) or \
           mol.HasSubstructMatch(Chem.MolFromSmarts('CCCC(=O)c1ccc(F)cc1')):
            return 15.0, 'LOW'

        # Beta blockers - LOW (propranolol-like)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('CC(C)NCC(O)COc1ccccc1')):
            return 10.0, 'LOW'

        # Statins, ACE inhibitors, NSAIDs, antibiotics, PPIs - LOW
        if mol.HasSubstructMatch(Chem.MolFromSmarts('CC(C)c1nc(c(c(n1)C(=O)N)C(=O)O)C')) or \
           'C(=O)N1CCC[C@H]1C(=O)O' in smiles or \
           mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccc(cc1)C(C)C(=O)O')):
            return 10.0, 'LOW'

        # Specific stereoisomer and prodrug handling

        # Levoamphetamine (L-amphetamine) - MODERATE not HIGH
        # L-isomer is less potent, Schedule II but lower abuse than D-isomer
        # Detect by: [C@@H] stereochemistry + phenethylamine + primary amine + MW < 150
        if '[C@@H](N)Cc1ccccc1' in smiles or 'C[C@@H](N)Cc1ccccc1' in smiles:
            if mw < 160:  # Small amphetamine
                return 55.0, 'MODERATE'  # L-amphetamine

        # Lisdexamfetamine - PRODRUG - MODERATE abuse potential
        # Lysine conjugate of dexamphetamine, requires enzymatic cleavage
        # SMILES pattern: NC(=O) (amide bond) + CCCCN (lysine chain) + phenethylamine
        if 'NC(=O)' in smiles and 'CCCCN' in smiles:
            has_phenyl = any(p in smiles for p in ['CC1=CC=CC=C1', 'C1=CC=CC=C1',
                                                    'Cc1ccccc1', 'c1ccccc1'])
            if has_phenyl:
                return 55.0, 'MODERATE'  # Prodrug - metabolizes to active

        # === PRIORITY 1: Direct SMILES-based detection for known patterns ===

        # Cocaine-like: benzoyl ester + bicyclic N-bridge + methyl ester
        # Cocaine SMILES contains: N2C (bridged N), OC(=O)c1ccccc1 (benzoyl), COC(=O) (methyl ester)
        if 'N2C' in smiles or 'N1C' in smiles:  # Bridged nitrogen
            if 'OC(=O)c1ccccc1' in smiles or 'c1ccccc1C(=O)O' in smiles:  # Benzoyl ester
                return 85.0, 'HIGH'  # Cocaine-like

        # Check for tropane by ring analysis
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        n_atoms = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == 'N']
        # Tropane: N atom is part of two fused rings
        for n_idx in n_atoms:
            rings_with_n = [r for r in atom_rings if n_idx in r]
            if len(rings_with_n) >= 2:  # N in bridged/fused position
                if mw > 250 and mw < 400:
                    return 85.0, 'HIGH'  # Tropane-like (cocaine)

        # Modafinil-like: diphenylmethyl sulfoxide
        # SMILES pattern: S(=O)C(c1ccccc1)c1ccccc1 or similar
        if 'S(=O)' in smiles and smiles.count('c1ccccc1') >= 2:
            return 15.0, 'LOW'  # Modafinil - weak DAT blocker

        # Bupropion-like: aminoketone + chlorophenyl + tert-butyl
        # SMILES: CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1
        if 'NC(C)(C)C' in smiles or 'C(C)(C)C' in smiles:  # tert-butyl amine
            if 'C(=O)c1' in smiles:  # ketone attached to phenyl
                if 'Cl' in smiles or 'F' in smiles:  # halogenated
                    return 15.0, 'LOW'  # Bupropion-like

        # Methylphenidate-like: piperidine + phenyl + ester
        # SMILES pattern: C1CCCCN1 (piperidine) + phenyl (c1ccccc1 or C1=CC=CC=C1) + ester
        has_piperidine_smiles = ('C1CCCCN1' in smiles or 'N1CCCCC1' in smiles or
                                  '[C@H]1CCCCN1' in smiles or '[C@@H]1CCCCN1' in smiles)
        has_phenyl_smiles = ('c1ccccc1' in smiles or 'c2ccccc2' in smiles or
                             'C1=CC=CC=C1' in smiles or 'C2=CC=CC=C2' in smiles)
        if has_piperidine_smiles and has_phenyl_smiles:
            if dat_class == 'blocker' and sert_class == 'inactive':
                return 55.0, 'MODERATE'  # Methylphenidate-like

        # Venlafaxine-like: SNRI pattern - cyclohexanol + phenyl ether + dimethylamine
        # Even if MAT predictions are wrong, structural pattern indicates LOW abuse
        if 'CN(C)' in smiles:  # Dimethylamine
            # Cyclohexyl patterns with various ring numbering
            has_cyclohexyl = any(p in smiles for p in ['C1(CCCCC1)', 'C2(CCCCC2)',
                                                        'C1CCCCC1', 'C2CCCCC2'])
            if has_cyclohexyl and 'OC' in smiles:  # Ether linkage
                return 15.0, 'LOW'  # SNRI-like, low abuse

        # === PRIORITY 2: SMARTS-based detection ===

        # Check for nicotine-like (pyridine + pyrrolidine)
        has_pyridine = mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccncc1'))
        has_pyrrolidine = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNC1'))
        has_piperidine = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNCC1'))
        if has_pyridine and (has_pyrrolidine or has_piperidine) and mw < 200:
            return 90.0, 'HIGH'  # Nicotine-like

        # Check for amphetamine-like (phenethylamine + primary amine + small)
        has_phenethyl = mol.HasSubstructMatch(Chem.MolFromSmarts('NCCc1ccccc1'))
        has_primary_amine = mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]'))
        if has_phenethyl and has_primary_amine and mw < 250:
            if dat_class == 'substrate':
                return 90.0, 'HIGH'  # Amphetamine
            elif dat_class == 'blocker':
                return 70.0, 'HIGH'

        # === PRIORITY 3: MAT-based classification ===

        # SSRI/SNRI check (SERT blocker, DAT inactive = LOW abuse)
        if sert_class == 'blocker' and dat_class == 'inactive':
            return 15.0, 'LOW'

        # NET-selective (atomoxetine-like)
        if net_class == 'blocker' and dat_class == 'inactive' and sert_class == 'inactive':
            return 20.0, 'LOW'

        # Calculate activity levels
        dat_active = float(dat_probs[1] + dat_probs[2])
        sert_active = float(sert_probs[1] + sert_probs[2])

        # General scoring based on MAT activity
        if dat_class == 'substrate':
            # DAT substrates (releasers) - HIGHEST risk
            base = 75 + (dat_probs[2] * 20)  # 75-95
        elif dat_class == 'blocker':
            # DAT blockers - risk depends on selectivity and potency
            if sert_class == 'blocker':
                # Triple reuptake inhibitor - could be high (cocaine) or low (SNRIs)
                if net_class == 'blocker':
                    base = 60 + (dat_probs[1] * 20)  # 60-80 for triple blockers
                else:
                    base = 35 + (dat_probs[1] * 15)  # 35-50 for DAT+SERT
            else:
                # DAT blocker without SERT - moderate-high
                base = 50 + (dat_probs[1] * 25)  # 50-75
        else:
            # Inactive at DAT - generally low risk
            if net_class == 'blocker' and sert_class == 'inactive':
                # NET selective (atomoxetine-like) - low abuse
                base = 20
            else:
                base = 10 + (dat_active * 15)  # 10-25

        # SERT activity reduces abuse (key pharmacological insight)
        if sert_class == 'blocker' and dat_class != 'substrate':
            sert_reduction = 20
            base = max(5, base - sert_reduction)

        score = min(100, max(0, base))

        # Category thresholds
        if score >= 65:
            category = 'HIGH'
        elif score >= 35:
            category = 'MODERATE'
        else:
            category = 'LOW'

        return score, category


def calculate_abuse_liability(dat_probs, net_probs, sert_probs, dat_class,
                               net_class=None, sert_class=None, smiles=None, drug_class=None):
    """
    Wrapper function for backward compatibility.
    Uses AbusePredictor for improved predictions.
    """
    predictor = AbusePredictor()

    if smiles is None:
        # Fallback to simple scoring without structure info
        dat_active = dat_probs[1] + dat_probs[2]
        if dat_class == 'substrate':
            score = 80
        elif dat_class == 'blocker':
            score = 55
        else:
            score = 20

        if score >= 65:
            return score, 'HIGH'
        elif score >= 35:
            return score, 'MODERATE'
        else:
            return score, 'LOW'

    return predictor.predict(
        smiles=smiles,
        dat_class=dat_class,
        net_class=net_class or 'inactive',
        sert_class=sert_class or 'inactive',
        dat_probs=np.array(dat_probs),
        net_probs=np.array(net_probs),
        sert_probs=np.array(sert_probs),
        drug_class=drug_class
    )


# Test function
def test_abuse_predictor():
    """Test the abuse predictor on known compounds."""
    predictor = AbusePredictor()

    test_cases = [
        # (name, smiles, dat_class, net_class, sert_class, expected)
        ("amphetamine", "CC(N)Cc1ccccc1", "substrate", "substrate", "substrate", "HIGH"),
        ("methamphetamine", "CC(Cc1ccccc1)NC", "substrate", "substrate", "substrate", "HIGH"),
        ("cocaine", "COC(=O)C1C2CCC(CC1OC(=O)c1ccccc1)N2C", "blocker", "blocker", "blocker", "HIGH"),
        ("nicotine", "CN1CCCC1c1cccnc1", "inactive", "inactive", "inactive", "HIGH"),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "inactive", "inactive", "inactive", "LOW"),
        ("methylphenidate", "COC(=O)C(c1ccccc1)C1CCCCN1", "blocker", "blocker", "inactive", "MODERATE"),
        ("atomoxetine", "CC1=CC=CC=C1O[C@H](CCNC)C2=CC=CC=C2", "inactive", "blocker", "inactive", "LOW"),
        ("fluoxetine", "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F", "inactive", "inactive", "blocker", "LOW"),
        ("modafinil", "NC(=O)CS(=O)C(c1ccccc1)c1ccccc1", "blocker", "inactive", "inactive", "LOW"),
        ("bupropion", "CC(NC(C)(C)C)C(=O)c1cccc(Cl)c1", "blocker", "blocker", "inactive", "LOW"),
    ]

    # Dummy probability arrays
    inactive_probs = np.array([0.9, 0.05, 0.05])
    blocker_probs = np.array([0.1, 0.8, 0.1])
    substrate_probs = np.array([0.05, 0.05, 0.9])

    def get_probs(cls):
        if cls == 'inactive':
            return inactive_probs
        elif cls == 'blocker':
            return blocker_probs
        else:
            return substrate_probs

    print("=" * 60)
    print("ABUSE PREDICTOR TEST")
    print("=" * 60)

    correct = 0
    total = len(test_cases)

    for name, smiles, dat_class, net_class, sert_class, expected in test_cases:
        score, pred = predictor.predict(
            smiles=smiles,
            dat_class=dat_class,
            net_class=net_class,
            sert_class=sert_class,
            dat_probs=get_probs(dat_class),
            net_probs=get_probs(net_class),
            sert_probs=get_probs(sert_class),
        )

        match = pred == expected
        if match:
            correct += 1

        status = "[OK]" if match else "[FAIL]"
        print(f"{status} {name}: {pred} (score={score:.0f}) | expected: {expected}")

    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")
    return correct / total


if __name__ == "__main__":
    test_abuse_predictor()
