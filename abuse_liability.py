"""
Abuse Liability Scoring Module
==============================

Converts StereoGNN predictions into abuse liability assessments
based on established pharmacological principles.

This is an INTERPRETATION layer - the GNN predicts mechanism,
this module applies pharmacology knowledge to infer risk.

References:
- Simmler et al. (2013) - Monoamine transporter interactions
- Baumann et al. (2012) - Designer drug pharmacology
- Heal et al. (2013) - ADHD medication abuse liability
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List, Tuple


class RiskLevel(Enum):
    """Abuse liability risk categories."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AbuseAssessment:
    """Complete abuse liability assessment."""
    score: int  # 0-100
    risk_level: RiskLevel
    primary_concern: str
    factors: List[Tuple[str, int, str]]  # (factor_name, points, explanation)
    mechanism_summary: str
    clinical_implications: str
    comparable_drugs: List[str]


class PharmacokineticsModifier:
    """
    Adjusts abuse liability based on pharmacokinetic factors.

    Key principle: Abuse potential depends not just on WHAT it does,
    but HOW FAST it does it.

    Fast onset = more reinforcing = higher abuse potential

    Examples:
    - Dexamphetamine vs Lisdexamfetamine: Same active drug, but
      lisdex requires enzymatic cleavage → slower onset → lower abuse
    - IR vs XR formulations: XR = slower onset = lower abuse
    """

    @staticmethod
    def detect_prodrug(smiles: str = "", name: str = "") -> Tuple[bool, str]:
        """
        Detect if compound is likely a prodrug.

        Returns:
            (is_prodrug, prodrug_type)
        """
        smiles_upper = smiles.upper() if smiles else ""
        name_lower = name.lower() if name else ""

        # Lysine conjugate pattern (lisdexamfetamine)
        # Lysine attachment: NCCCC[C@H](N)C(=O)N-drug
        if "NCCCC" in smiles_upper and "C(=O)N" in smiles_upper:
            return True, "lysine_prodrug"

        # Name-based detection
        if "lisdex" in name_lower or "lis-" in name_lower:
            return True, "lysine_prodrug"
        if "vyvanse" in name_lower:
            return True, "lysine_prodrug"
        if "prodrug" in name_lower:
            return True, "generic_prodrug"

        # Ester prodrug patterns (less common for stimulants)
        if name_lower.endswith("pivalate") or name_lower.endswith("propionate"):
            return True, "ester_prodrug"

        return False, ""

    @staticmethod
    def get_formulation_modifier(formulation: str = "ir") -> Tuple[float, str]:
        """
        Get abuse liability modifier based on formulation.

        Args:
            formulation: 'ir' (immediate release), 'xr'/'er' (extended),
                        'patch' (transdermal), 'prodrug'

        Returns:
            (multiplier, explanation)
        """
        formulation = formulation.lower()

        modifiers = {
            'ir': (1.0, "Immediate release - standard abuse potential"),
            'xr': (0.7, "Extended release - slower onset, reduced abuse potential"),
            'er': (0.7, "Extended release - slower onset, reduced abuse potential"),
            'sr': (0.75, "Sustained release - moderately reduced abuse potential"),
            'patch': (0.5, "Transdermal - very slow onset, significantly reduced abuse potential"),
            'prodrug': (0.5, "Prodrug - requires metabolic activation, reduced abuse potential"),
            'iv': (1.4, "IV formulation - immediate onset, highest abuse potential"),
        }

        return modifiers.get(formulation, (1.0, "Unknown formulation"))

    @staticmethod
    def calculate_onset_score(
        is_prodrug: bool = False,
        prodrug_type: str = "",
        formulation: str = "ir"
    ) -> Tuple[float, List[str]]:
        """
        Calculate overall PK modifier for abuse liability.

        Returns:
            (multiplier, list of explanations)
        """
        multiplier = 1.0
        explanations = []

        # Prodrug adjustment
        if is_prodrug:
            if prodrug_type == "lysine_prodrug":
                multiplier *= 0.5
                explanations.append(
                    "Lysine prodrug: requires enzymatic cleavage in RBCs, "
                    "~1-2hr delayed onset, cannot be effectively snorted/injected"
                )
            else:
                multiplier *= 0.7
                explanations.append("Prodrug formulation: delayed onset")

        # Formulation adjustment (only if not already prodrug)
        if not is_prodrug:
            form_mult, form_expl = PharmacokineticsModifier.get_formulation_modifier(formulation)
            if form_mult != 1.0:
                multiplier *= form_mult
                explanations.append(form_expl)

        return multiplier, explanations


class AbuseLiabilityScorer:
    """
    Scores abuse liability based on transporter pharmacology.

    Pharmacological Basis:
    ----------------------
    1. DAT substrates (releasers) > DAT blockers for abuse potential
       - Releasers cause rapid, massive dopamine release
       - Blockers cause gradual accumulation

    2. DAT activity correlates with reinforcement/reward
       - Dopamine = primary reward neurotransmitter
       - Higher DAT potency = stronger euphoria

    3. Speed of onset matters
       - Fast DAT binding = more abusable
       - pKi correlates with binding speed

    4. Selectivity patterns:
       - DAT-preferring: stimulant abuse (meth, cocaine)
       - SERT-preferring: less abuse potential (SSRIs)
       - Non-selective: unpredictable, potentially dangerous
    """

    # Scoring weights based on pharmacology literature
    WEIGHTS = {
        # Mechanism points (substrate vs blocker)
        'dat_substrate': 35,
        'dat_blocker': 15,
        'net_substrate': 10,
        'net_blocker': 5,
        'sert_substrate': 8,
        'sert_blocker': 3,

        # Potency thresholds (pKi)
        'pki_very_high': 25,    # pKi >= 8
        'pki_high': 18,         # pKi >= 7
        'pki_moderate': 10,     # pKi >= 6
        'pki_low': 3,           # pKi >= 5

        # Selectivity bonuses/penalties
        'dat_preferring': 12,   # DAT >> SERT
        'sert_preferring': -10, # SERT >> DAT (protective)
        'non_selective': 8,     # Hits everything

        # Kinetic factors
        'fast_onset': 10,       # High kon
        'releaser_kinetics': 8, # Substrate with high efficacy
    }

    # Risk level thresholds
    THRESHOLDS = {
        RiskLevel.MINIMAL: (0, 15),
        RiskLevel.LOW: (16, 35),
        RiskLevel.MODERATE: (36, 55),
        RiskLevel.HIGH: (56, 75),
        RiskLevel.VERY_HIGH: (76, 100),
    }

    # Reference compounds for comparison
    REFERENCE_COMPOUNDS = {
        (0, 15): ["Caffeine", "Modafinil"],
        (16, 35): ["Bupropion", "Atomoxetine"],
        (36, 55): ["Methylphenidate", "Cocaine (low dose)"],
        (56, 75): ["Amphetamine", "Cocaine", "MDMA"],
        (76, 100): ["Methamphetamine", "Mephedrone", "MDPV"],
    }

    def score(
        self,
        prediction: Dict,
        smiles: str = "",
        name: str = "",
        formulation: str = "ir"
    ) -> AbuseAssessment:
        """
        Calculate abuse liability from model prediction.

        Args:
            prediction: Dict with keys like:
                - 'DAT': 'substrate' | 'blocker' | 'inactive'
                - 'NET': 'substrate' | 'blocker' | 'inactive'
                - 'SERT': 'substrate' | 'blocker' | 'inactive'
                - 'DAT_pKi': float (optional)
                - 'NET_pKi': float (optional)
                - 'SERT_pKi': float (optional)
            smiles: SMILES string (for prodrug detection)
            name: Drug name (for prodrug detection)
            formulation: 'ir', 'xr', 'er', 'patch', etc.

        Returns:
            AbuseAssessment with score, risk level, and explanation
        """
        factors = []
        total_score = 0

        # --- Check for prodrug/formulation modifiers ---
        is_prodrug, prodrug_type = PharmacokineticsModifier.detect_prodrug(smiles, name)
        pk_multiplier, pk_explanations = PharmacokineticsModifier.calculate_onset_score(
            is_prodrug, prodrug_type, formulation
        )

        # --- Mechanism Scoring ---

        # DAT (most important for abuse)
        dat_status = prediction.get('DAT', 'inactive').lower()
        if dat_status == 'substrate':
            points = self.WEIGHTS['dat_substrate']
            factors.append(('DAT Substrate', points,
                'Dopamine releaser - causes rapid DA efflux, strong reinforcement'))
            total_score += points
        elif dat_status == 'blocker':
            points = self.WEIGHTS['dat_blocker']
            factors.append(('DAT Blocker', points,
                'Dopamine reuptake inhibitor - accumulation without release'))
            total_score += points

        # NET
        net_status = prediction.get('NET', 'inactive').lower()
        if net_status == 'substrate':
            points = self.WEIGHTS['net_substrate']
            factors.append(('NET Substrate', points,
                'Norepinephrine releaser - contributes to stimulant effects'))
            total_score += points
        elif net_status == 'blocker':
            points = self.WEIGHTS['net_blocker']
            factors.append(('NET Blocker', points,
                'NE reuptake inhibitor - alertness without strong reward'))
            total_score += points

        # SERT
        sert_status = prediction.get('SERT', 'inactive').lower()
        if sert_status == 'substrate':
            points = self.WEIGHTS['sert_substrate']
            factors.append(('SERT Substrate', points,
                'Serotonin releaser - entactogenic effects, hyperthermia risk'))
            total_score += points
        elif sert_status == 'blocker':
            points = self.WEIGHTS['sert_blocker']
            factors.append(('SERT Blocker', points,
                'Serotonin reuptake inhibitor - mood effects'))
            total_score += points

        # --- Potency Scoring (DAT is key) ---

        dat_pki = prediction.get('DAT_pKi') or prediction.get('pKi_DAT')
        if dat_pki is not None and dat_status != 'inactive':
            if dat_pki >= 8:
                points = self.WEIGHTS['pki_very_high']
                factors.append(('Very High DAT Potency', points,
                    f'pKi={dat_pki:.1f} - extremely potent, rapid onset'))
            elif dat_pki >= 7:
                points = self.WEIGHTS['pki_high']
                factors.append(('High DAT Potency', points,
                    f'pKi={dat_pki:.1f} - potent binding'))
            elif dat_pki >= 6:
                points = self.WEIGHTS['pki_moderate']
                factors.append(('Moderate DAT Potency', points,
                    f'pKi={dat_pki:.1f} - moderate binding'))
            elif dat_pki >= 5:
                points = self.WEIGHTS['pki_low']
                factors.append(('Low DAT Potency', points,
                    f'pKi={dat_pki:.1f} - weak binding'))
            total_score += points

        # --- Selectivity Scoring ---

        sert_pki = prediction.get('SERT_pKi') or prediction.get('pKi_SERT')

        if dat_pki and sert_pki:
            selectivity = dat_pki - sert_pki
            if selectivity > 1.0:  # DAT-preferring
                points = self.WEIGHTS['dat_preferring']
                factors.append(('DAT-Preferring', points,
                    f'DAT/SERT ratio favors dopamine - classic stimulant profile'))
                total_score += points
            elif selectivity < -1.0:  # SERT-preferring
                points = self.WEIGHTS['sert_preferring']
                factors.append(('SERT-Preferring', points,
                    f'SERT/DAT ratio favors serotonin - reduced abuse potential'))
                total_score += points

        # Non-selective (hits all three as substrates)
        if (dat_status == 'substrate' and
            net_status == 'substrate' and
            sert_status == 'substrate'):
            points = self.WEIGHTS['non_selective']
            factors.append(('Non-Selective Releaser', points,
                'Releases DA, NE, and 5-HT - unpredictable effects, higher risk'))
            total_score += points

        # --- Apply PK modifier (prodrug/formulation) ---
        if pk_multiplier != 1.0:
            original_score = total_score
            total_score = int(total_score * pk_multiplier)
            reduction = original_score - total_score
            for expl in pk_explanations:
                factors.append(('Pharmacokinetic Modifier', -reduction, expl))

        # --- Cap score at 100 ---
        total_score = min(100, max(0, total_score))

        # --- Determine risk level ---
        risk_level = RiskLevel.MINIMAL
        for level, (low, high) in self.THRESHOLDS.items():
            if low <= total_score <= high:
                risk_level = level
                break

        # --- Generate summaries ---
        mechanism_summary = self._generate_mechanism_summary(prediction, is_prodrug, prodrug_type)
        clinical_implications = self._generate_clinical_implications(
            prediction, risk_level, factors)
        comparable_drugs = self._get_comparable_drugs(total_score)
        primary_concern = self._get_primary_concern(factors)

        return AbuseAssessment(
            score=total_score,
            risk_level=risk_level,
            primary_concern=primary_concern,
            factors=factors,
            mechanism_summary=mechanism_summary,
            clinical_implications=clinical_implications,
            comparable_drugs=comparable_drugs,
        )

    def _generate_mechanism_summary(
        self,
        prediction: Dict,
        is_prodrug: bool = False,
        prodrug_type: str = ""
    ) -> str:
        """Generate human-readable mechanism summary."""
        parts = []

        for target in ['DAT', 'NET', 'SERT']:
            status = prediction.get(target, 'inactive').lower()
            if status == 'substrate':
                parts.append(f"{target} releaser")
            elif status == 'blocker':
                parts.append(f"{target} blocker")

        if not parts:
            return "No significant monoamine transporter activity predicted."

        summary = "Predicted mechanism: " + ", ".join(parts) + "."

        if is_prodrug:
            if prodrug_type == "lysine_prodrug":
                summary += " NOTE: Lysine prodrug - active drug released slowly after oral absorption."
            else:
                summary += " NOTE: Prodrug formulation - requires metabolic activation."

        return summary

    def _generate_clinical_implications(
        self,
        prediction: Dict,
        risk_level: RiskLevel,
        factors: List
    ) -> str:
        """Generate clinical interpretation."""

        dat_status = prediction.get('DAT', 'inactive').lower()
        sert_status = prediction.get('SERT', 'inactive').lower()

        implications = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            implications.append("HIGH ABUSE LIABILITY - Schedule II equivalent profile.")

        if dat_status == 'substrate':
            implications.append(
                "Dopamine releasing agent: expect euphoria, reinforcement, "
                "cardiovascular stimulation, potential for compulsive redosing.")
        elif dat_status == 'blocker':
            implications.append(
                "Dopamine reuptake inhibitor: stimulant effects without "
                "the intense rush of releasing agents.")

        if sert_status == 'substrate':
            implications.append(
                "Serotonin releaser: risk of hyperthermia, serotonin syndrome "
                "with other serotonergics. Entactogenic properties likely.")

        if not implications:
            implications.append("Limited monoamine activity - low CNS stimulant potential.")

        return " ".join(implications)

    def _get_comparable_drugs(self, score: int) -> List[str]:
        """Get reference drugs with similar profiles."""
        for (low, high), drugs in self.REFERENCE_COMPOUNDS.items():
            if low <= score <= high:
                return drugs
        return ["Unknown profile"]

    def _get_primary_concern(self, factors: List) -> str:
        """Identify the main risk factor."""
        if not factors:
            return "No significant concerns identified"
        # Return the highest-scoring factor
        sorted_factors = sorted(factors, key=lambda x: x[1], reverse=True)
        return sorted_factors[0][0]


def score_prediction(prediction: Dict) -> AbuseAssessment:
    """Convenience function for scoring a single prediction."""
    scorer = AbuseLiabilityScorer()
    return scorer.score(prediction)


def format_assessment(assessment: AbuseAssessment) -> str:
    """Format assessment as readable text report."""
    lines = [
        "=" * 60,
        "ABUSE LIABILITY ASSESSMENT",
        "=" * 60,
        "",
        f"SCORE: {assessment.score}/100",
        f"RISK LEVEL: {assessment.risk_level.value.upper()}",
        f"PRIMARY CONCERN: {assessment.primary_concern}",
        "",
        "-" * 60,
        "MECHANISM",
        "-" * 60,
        assessment.mechanism_summary,
        "",
        "-" * 60,
        "SCORING FACTORS",
        "-" * 60,
    ]

    for factor_name, points, explanation in assessment.factors:
        lines.append(f"  +{points:2d}  {factor_name}")
        lines.append(f"       {explanation}")

    lines.extend([
        "",
        "-" * 60,
        "CLINICAL IMPLICATIONS",
        "-" * 60,
        assessment.clinical_implications,
        "",
        "-" * 60,
        "COMPARABLE DRUGS",
        "-" * 60,
        "  " + ", ".join(assessment.comparable_drugs),
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


# === Example Usage ===

if __name__ == "__main__":
    scorer = AbuseLiabilityScorer()

    # Example 1: Methamphetamine-like profile
    meth_profile = {
        'DAT': 'substrate',
        'NET': 'substrate',
        'SERT': 'substrate',
        'DAT_pKi': 7.8,
        'NET_pKi': 7.2,
        'SERT_pKi': 6.5,
    }

    print("\n=== METHAMPHETAMINE-LIKE COMPOUND ===")
    result = scorer.score(meth_profile)
    print(format_assessment(result))

    # Example 2: SSRI-like profile
    ssri_profile = {
        'DAT': 'inactive',
        'NET': 'inactive',
        'SERT': 'blocker',
        'SERT_pKi': 8.5,
    }

    print("\n=== SSRI-LIKE COMPOUND ===")
    result = scorer.score(ssri_profile)
    print(format_assessment(result))

    # Example 3: Methylphenidate-like profile
    mph_profile = {
        'DAT': 'blocker',
        'NET': 'blocker',
        'SERT': 'inactive',
        'DAT_pKi': 7.0,
        'NET_pKi': 6.8,
    }

    print("\n=== METHYLPHENIDATE-LIKE COMPOUND ===")
    result = scorer.score(mph_profile)
    print(format_assessment(result))

    # Example 4: MDMA-like profile
    mdma_profile = {
        'DAT': 'substrate',
        'NET': 'substrate',
        'SERT': 'substrate',
        'DAT_pKi': 6.2,
        'NET_pKi': 6.5,
        'SERT_pKi': 7.8,
    }

    print("\n=== MDMA-LIKE COMPOUND ===")
    result = scorer.score(mdma_profile)
    print(format_assessment(result))

    # Example 5: Dexamphetamine vs Lisdexamfetamine comparison
    # Same pharmacology, different PK = different abuse potential

    dex_profile = {
        'DAT': 'substrate',
        'NET': 'substrate',
        'SERT': 'inactive',
        'DAT_pKi': 7.1,
        'NET_pKi': 6.8,
    }

    print("\n" + "=" * 60)
    print("DEXAMPHETAMINE vs LISDEXAMFETAMINE COMPARISON")
    print("=" * 60)

    print("\n=== DEXAMPHETAMINE (IR) ===")
    result_dex = scorer.score(dex_profile, name="dexamphetamine", formulation="ir")
    print(format_assessment(result_dex))

    print("\n=== DEXAMPHETAMINE (XR) ===")
    result_dex_xr = scorer.score(dex_profile, name="dexamphetamine", formulation="xr")
    print(format_assessment(result_dex_xr))

    print("\n=== LISDEXAMFETAMINE (Vyvanse) ===")
    result_lisdex = scorer.score(dex_profile, name="lisdexamfetamine", formulation="ir")
    print(format_assessment(result_lisdex))

    print("\n" + "=" * 60)
    print("SUMMARY: Same active drug, different abuse liability")
    print("=" * 60)
    print(f"  Dexamphetamine IR:    {result_dex.score}/100 ({result_dex.risk_level.value})")
    print(f"  Dexamphetamine XR:    {result_dex_xr.score}/100 ({result_dex_xr.risk_level.value})")
    print(f"  Lisdexamfetamine:     {result_lisdex.score}/100 ({result_lisdex.risk_level.value})")
    print("=" * 60)
