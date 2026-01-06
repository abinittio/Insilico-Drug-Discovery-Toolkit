"""
Pharmacology Rule Engine
========================
Post-processing rules to correct model predictions based on known pharmacology.

This applies established structure-activity relationships and drug class rules
to improve prediction accuracy, especially for substrate vs blocker distinction.
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import Dict, Optional, Tuple
import re


class PharmacologyRules:
    """
    Rule-based corrections for MAT predictions.

    Key pharmacological principles:
    1. Primary amines with phenethylamine scaffold → likely DAT/NET substrate
    2. Secondary/tertiary amines, larger molecules → likely blocker
    3. SSRIs have specific structural features → SERT blocker, not substrate
    4. Tropane alkaloids (cocaine-like) → triple blocker
    """

    # Known drug class patterns (SMARTS)
    PATTERNS = {
        # Phenethylamine core: C-C-N attached to phenyl
        'phenethylamine': '[CH2][CH]([NH2,NH3+])c1ccccc1',
        'phenethylamine_substituted': '[CH2][CH]([NH])c1ccccc1',

        # Amphetamine-like: alpha-methyl phenethylamine
        'amphetamine_core': 'CC(N)Cc1ccccc1',
        'amphetamine_substituted': 'CC(NC)Cc1ccccc1',

        # Cathinone (beta-keto amphetamine)
        'cathinone': 'CC(N)C(=O)c1ccccc1',

        # Tropane (cocaine-like)
        'tropane': 'C1CC2CCC1N2',

        # Piperidine with phenyl (methylphenidate-like)
        'piperidine_phenyl': 'C1CCNCC1c2ccccc2',

        # Diphenylmethyl (modafinil-like)
        'diphenylmethyl': 'C(c1ccccc1)c2ccccc2',

        # SSRI-like: propylamine chain with aryl ether
        'ssri_core': 'NCCC(*)Oc1ccc(**)cc1',

        # Tricyclic core
        'tricyclic': 'c1ccc2c(c1)CCc3ccccc3N2',
    }

    # Known drug classes with expected profiles
    DRUG_CLASS_PROFILES = {
        'amphetamine': {
            'DAT': 'substrate', 'NET': 'substrate', 'SERT': 'weak_substrate',
            'abuse': 'HIGH', 'mechanism': 'Releases monoamines via reverse transport'
        },
        'methylphenidate': {
            'DAT': 'blocker', 'NET': 'blocker', 'SERT': 'inactive',
            'abuse': 'MODERATE', 'mechanism': 'Reuptake inhibitor, not releaser'
        },
        'cocaine': {
            'DAT': 'blocker', 'NET': 'blocker', 'SERT': 'blocker',
            'abuse': 'HIGH', 'mechanism': 'Triple reuptake inhibitor'
        },
        'ssri': {
            'DAT': 'inactive', 'NET': 'inactive', 'SERT': 'blocker',
            'abuse': 'LOW', 'mechanism': 'Selective serotonin reuptake inhibitor'
        },
        'snri': {
            'DAT': 'inactive', 'NET': 'blocker', 'SERT': 'blocker',
            'abuse': 'LOW', 'mechanism': 'Serotonin-norepinephrine reuptake inhibitor'
        },
        'nri': {
            'DAT': 'inactive', 'NET': 'blocker', 'SERT': 'inactive',
            'abuse': 'LOW', 'mechanism': 'Selective norepinephrine reuptake inhibitor'
        },
        'cathinone': {
            'DAT': 'substrate', 'NET': 'substrate', 'SERT': 'substrate',
            'abuse': 'HIGH', 'mechanism': 'Synthetic cathinone, releaser'
        },
    }

    def __init__(self):
        # Compile SMARTS patterns
        self.compiled_patterns = {}
        for name, smarts in self.PATTERNS.items():
            try:
                self.compiled_patterns[name] = Chem.MolFromSmarts(smarts)
            except:
                pass

    def classify_structure(self, smiles: str) -> Dict:
        """
        Classify molecule based on structural features.
        Returns detected patterns and likely drug class.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}

        result = {
            'patterns_matched': [],
            'likely_class': None,
            'confidence': 'low',
            'features': {}
        }

        # Check patterns
        for name, pattern in self.compiled_patterns.items():
            if pattern and mol.HasSubstructMatch(pattern):
                result['patterns_matched'].append(name)

        # Calculate relevant descriptors
        result['features'] = {
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_amines': self._count_amines(mol),
            'primary_amine': self._has_primary_amine(mol),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        }

        # Classify based on patterns and features
        result['likely_class'] = self._determine_class(result)

        return result

    def _count_amines(self, mol) -> int:
        """Count nitrogen atoms that are likely amines."""
        amine_pattern = Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]')
        if amine_pattern:
            return len(mol.GetSubstructMatches(amine_pattern))
        return 0

    def _has_primary_amine(self, mol) -> bool:
        """Check for primary amine (NH2)."""
        primary = Chem.MolFromSmarts('[NH2]')
        return mol.HasSubstructMatch(primary) if primary else False

    def _determine_class(self, analysis: Dict) -> Optional[str]:
        """Determine likely drug class from structural analysis."""
        patterns = analysis['patterns_matched']
        features = analysis['features']

        # Amphetamine class: phenethylamine + primary amine + small
        if ('amphetamine_core' in patterns or 'phenethylamine' in patterns):
            if features['primary_amine'] and features['mol_weight'] < 250:
                return 'amphetamine'

        # Substituted amphetamine (methamphetamine, MDMA)
        if 'amphetamine_substituted' in patterns:
            if features['mol_weight'] < 300:
                return 'amphetamine'

        # Cathinone
        if 'cathinone' in patterns:
            return 'cathinone'

        # Tropane (cocaine-like)
        if 'tropane' in patterns:
            return 'cocaine'

        # Methylphenidate-like: piperidine + phenyl + ester
        if 'piperidine_phenyl' in patterns:
            if features['mol_weight'] < 350:
                return 'methylphenidate'

        # SSRI-like: larger, no primary amine, aromatic ethers
        if features['mol_weight'] > 250 and not features['primary_amine']:
            if features['num_aromatic_rings'] >= 2:
                # Check for fluorine (common in SSRIs)
                if 'F' in analysis.get('smiles', ''):
                    return 'ssri'

        return None

    def correct_predictions(self, smiles: str, model_predictions: Dict) -> Dict:
        """
        Apply pharmacological rules to correct model predictions.

        Args:
            smiles: Input molecule SMILES
            model_predictions: Dict with DAT_class, NET_class, SERT_class, etc.

        Returns:
            Corrected predictions with confidence and explanation
        """
        analysis = self.classify_structure(smiles)
        corrected = model_predictions.copy()
        corrections_made = []

        likely_class = analysis.get('likely_class')

        if likely_class and likely_class in self.DRUG_CLASS_PROFILES:
            profile = self.DRUG_CLASS_PROFILES[likely_class]

            # Apply corrections based on drug class
            for target in ['DAT', 'NET', 'SERT']:
                model_pred = model_predictions.get(f'{target}_class')
                expected = profile[target]

                # Convert weak_substrate to substrate for comparison
                expected_simple = 'substrate' if 'substrate' in expected else expected

                if model_pred != expected_simple:
                    # Check if this is a blocker/substrate confusion
                    if model_pred in ['blocker', 'substrate'] and expected_simple in ['blocker', 'substrate']:
                        corrected[f'{target}_class'] = expected_simple
                        corrected[f'{target}_corrected'] = True
                        corrections_made.append(
                            f"{target}: {model_pred} → {expected_simple} (rule: {likely_class} class)"
                        )
                    # Check if model said inactive but should be active
                    elif model_pred == 'inactive' and expected_simple != 'inactive':
                        corrected[f'{target}_class'] = expected_simple
                        corrected[f'{target}_corrected'] = True
                        corrections_made.append(
                            f"{target}: inactive → {expected_simple} (rule: {likely_class} class)"
                        )

            # Correct abuse prediction
            if 'abuse_category' in model_predictions:
                if model_predictions['abuse_category'] != profile['abuse']:
                    corrected['abuse_category'] = profile['abuse']
                    corrected['abuse_corrected'] = True
                    corrections_made.append(
                        f"Abuse: {model_predictions['abuse_category']} → {profile['abuse']}"
                    )

        # Additional rule: Primary amine phenethylamines are almost always substrates
        if analysis['features'].get('primary_amine') and 'phenethylamine' in analysis.get('patterns_matched', []):
            for target in ['DAT', 'NET']:
                if corrected.get(f'{target}_class') == 'blocker':
                    corrected[f'{target}_class'] = 'substrate'
                    corrected[f'{target}_corrected'] = True
                    corrections_made.append(
                        f"{target}: blocker → substrate (rule: primary amine phenethylamine)"
                    )

        corrected['analysis'] = analysis
        corrected['corrections'] = corrections_made
        corrected['drug_class'] = likely_class

        return corrected

    def get_explanation(self, corrected: Dict) -> str:
        """Generate human-readable explanation of corrections."""
        lines = []

        if corrected.get('drug_class'):
            profile = self.DRUG_CLASS_PROFILES.get(corrected['drug_class'], {})
            lines.append(f"Detected drug class: {corrected['drug_class'].upper()}")
            if 'mechanism' in profile:
                lines.append(f"Mechanism: {profile['mechanism']}")

        if corrected.get('corrections'):
            lines.append("\nCorrections applied:")
            for c in corrected['corrections']:
                lines.append(f"  • {c}")

        patterns = corrected.get('analysis', {}).get('patterns_matched', [])
        if patterns:
            lines.append(f"\nStructural features: {', '.join(patterns)}")

        return '\n'.join(lines)


def apply_rules(smiles: str, predictions: Dict) -> Dict:
    """Convenience function to apply pharmacology rules."""
    rules = PharmacologyRules()
    return rules.correct_predictions(smiles, predictions)


# Test
if __name__ == "__main__":
    rules = PharmacologyRules()

    test_cases = [
        ("CC(N)Cc1ccccc1", "Amphetamine"),
        ("CC(Cc1ccccc1)NC", "Methamphetamine"),
        ("COC(=O)C(c1ccccc1)C1CCCCN1", "Methylphenidate"),
        ("COC(=O)C1C2CCC(CC1OC(=O)c1ccccc1)N2C", "Cocaine"),
        ("CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "Fluoxetine"),
    ]

    for smiles, name in test_cases:
        print(f"\n{'='*50}")
        print(f"{name}: {smiles}")
        analysis = rules.classify_structure(smiles)
        print(f"Detected class: {analysis.get('likely_class')}")
        print(f"Patterns: {analysis.get('patterns_matched')}")
        print(f"Primary amine: {analysis['features'].get('primary_amine')}")
        print(f"MW: {analysis['features'].get('mol_weight'):.1f}")
