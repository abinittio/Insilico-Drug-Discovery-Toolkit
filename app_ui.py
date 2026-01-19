"""
Insilico Drug Discovery Toolkit
================================
World-class UI for SOTA molecular property prediction.

Run: streamlit run app_ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, DataStructs
import io
import base64
import requests
import urllib.parse

# Cache for PubChem lookups
_smiles_cache = {}


def name_to_smiles(name: str) -> str:
    """Convert any chemical/drug name to SMILES using PubChem API."""
    name = name.strip()
    if not name:
        return None

    # Check cache first
    cache_key = name.lower()
    if cache_key in _smiles_cache:
        return _smiles_cache[cache_key]

    # Check if it's already a valid SMILES
    mol = Chem.MolFromSmiles(name)
    if mol is not None:
        return name

    # Query PubChem
    try:
        encoded_name = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IsomericSMILES/JSON"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            props = data['PropertyTable']['Properties'][0]
            # Try different SMILES field names (PubChem API is inconsistent)
            smiles = props.get('SMILES') or props.get('IsomericSMILES') or props.get('CanonicalSMILES')
            if smiles:
                _smiles_cache[cache_key] = smiles
                return smiles
    except Exception as e:
        print(f"PubChem lookup failed for '{name}': {e}")

    return None


def apply_pharmacology_rules(smiles: str, raw_preds: dict) -> dict:
    """
    Apply pharmacological rules to correct model predictions.

    Key corrections:
    1. Known inactive drug classes -> force inactive
    2. Low confidence + no MAT features -> likely inactive
    3. Known SSRI/SNRI patterns -> correct mechanism
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return raw_preds

    result = {t: dict(raw_preds[t]) for t in ['DAT', 'NET', 'SERT']}
    result['drug_class'] = None

    # Calculate features
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    has_primary_amine = mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]'))
    has_phenethylamine = mol.HasSubstructMatch(Chem.MolFromSmarts('NCCc1ccccc1'))
    has_cathinone = mol.HasSubstructMatch(Chem.MolFromSmarts('NC(C)C(=O)c1ccccc1'))
    has_tropane = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CC2CCC1N2'))
    has_piperazine = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CNCCN1'))
    has_sulfonamide = mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N'))
    has_carbamate = mol.HasSubstructMatch(Chem.MolFromSmarts('NC(=O)O'))
    has_pyrrolidine = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNC1'))
    has_piperidine_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNCC1'))
    has_amide = mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)N'))  # Amides are blockers not releasers
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    # RULE 1: Known inactive drug classes (anticonvulsants, GABAergics)
    # High MW, no basic nitrogen near aromatic, specific substructures
    inactive_patterns = [
        Chem.MolFromSmarts('S(=O)(=O)NC(=O)'),  # Sulfonamide anticonvulsants
        Chem.MolFromSmarts('C1CC(=O)NC(=O)C1'),  # Barbiturate-like
        Chem.MolFromSmarts('C1C(N)C(=O)NC1=O'),  # Hydantoin
    ]

    is_likely_inactive = False

    # Gabapentinoid pattern (cyclohexane + amino acid)
    if mol.HasSubstructMatch(Chem.MolFromSmarts('NCC(=O)O')) and not has_phenethylamine:
        is_likely_inactive = True
        result['drug_class'] = 'gabapentinoid'

    # Sulfonamide drugs (topiramate-like)
    if has_sulfonamide and not has_phenethylamine:
        is_likely_inactive = True
        result['drug_class'] = 'sulfonamide'

    # Very high TPSA suggests not CNS active at transporters
    if tpsa > 120 and not has_phenethylamine and not has_tropane:
        is_likely_inactive = True

    # RULE 2: Amphetamine substrates (strict criteria)
    # Must have: phenethylamine + primary amine + small + no extra complex rings
    # Exclude: amides (solriamfetol), sulfonamides, pyrrolidines, piperidines -> these are blockers

    if has_phenethylamine and has_primary_amine and mw < 220 and ring_count <= 2:
        # These modifications convert substrates to blockers
        if not has_pyrrolidine and not has_piperidine_ring and not has_amide and not has_sulfonamide:
            result['drug_class'] = 'amphetamine'
            for t in ['DAT', 'NET']:
                if result[t]['class'] != 'substrate':
                    result[t]['class'] = 'substrate'
                    result[t]['corrected'] = True

    # Cathinone substrates - but NOT pyrrolidine cathinones (those are blockers)
    if has_cathinone and not has_pyrrolidine:
        result['drug_class'] = 'cathinone'
        for t in ['DAT', 'NET', 'SERT']:
            if result[t]['class'] != 'substrate':
                result[t]['class'] = 'substrate'
                result[t]['corrected'] = True

    # Pyrrolidine cathinones are BLOCKERS (alpha-PVP, MDPV)
    if has_cathinone and has_pyrrolidine:
        result['drug_class'] = 'pyrovalerone'
        for t in ['DAT', 'NET']:
            if result[t]['class'] != 'blocker':
                result[t]['class'] = 'blocker'
                result[t]['corrected'] = True
        # SERT usually inactive for these
        if result['SERT']['class'] != 'inactive':
            result['SERT']['class'] = 'inactive'
            result['SERT']['corrected'] = True

    # RULE 3: Tropane = blocker (cocaine-like)
    if has_tropane:
        result['drug_class'] = 'tropane'
        for t in ['DAT', 'NET']:
            if result[t]['class'] == 'inactive':
                result[t]['class'] = 'blocker'
                result[t]['corrected'] = True

    # RULE 4: Force inactive for known inactive patterns
    if is_likely_inactive:
        for t in ['DAT', 'NET', 'SERT']:
            if result[t]['class'] != 'inactive':
                result[t]['class'] = 'inactive'
                result[t]['corrected'] = True

    # RULE 5: Low confidence inactive bias
    # If model is uncertain (max_prob < 0.6) and no clear MAT features, default to inactive
    for t in ['DAT', 'NET', 'SERT']:
        if result[t]['max_prob'] < 0.55 and not has_phenethylamine and not has_tropane:
            if result[t]['class'] != 'inactive':
                result[t]['class'] = 'inactive'
                result[t]['corrected'] = True

    return result


def resolve_input(user_input: str) -> tuple:
    """Convert chemical name to SMILES. Returns (smiles, was_converted, original_name)."""
    user_input = user_input.strip()

    # Try to parse as SMILES first
    mol = Chem.MolFromSmiles(user_input)
    if mol is not None:
        return user_input, False, None

    # Try PubChem lookup
    smiles = name_to_smiles(user_input)
    if smiles:
        return smiles, True, user_input

    # Return original input (will fail validation later)
    return user_input, False, None


# Page config - must be first
st.set_page_config(
    page_title="Insilico Drug Discovery Toolkit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Blue theme with proper contrast
st.markdown("""
<style>
    /* Main background - dark blue gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #101025 50%, #0d1520 100%);
    }

    /* Headers - bright blue/cyan gradient */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
    }

    h2, h3 {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
        font-weight: 600;
    }

    /* ALL text - white/light for contrast */
    p, span, div, label, .stMarkdown {
        color: #e8e8e8 !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #151525, #1a1a30);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(96,165,250,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        margin: 10px 0;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #a0a0a0 !important;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status badges */
    .badge-safe {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: white !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-warning {
        background: linear-gradient(135deg, #ff9800, #ffb74d);
        color: black !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-danger {
        background: linear-gradient(135deg, #f44336, #e57373);
        color: white !important;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: #151525 !important;
        border: 2px solid #60a5fa !important;
        border-radius: 12px;
        color: white !important;
        font-size: 1.1rem;
        padding: 12px 16px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0,212,255,0.4);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.5);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a0a15 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Tables - white text */
    .stDataFrame {
        background: #151525 !important;
    }

    table {
        color: #e8e8e8 !important;
    }

    th {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
        font-weight: 600;
        background: #1a1a30 !important;
    }

    td {
        color: #e8e8e8 !important;
        background: #151525 !important;
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #7c3aed) !important;
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
        font-weight: 600;
    }

    /* Info boxes */
    .stAlert {
        background: #1a1a30 !important;
        color: #e8e8e8 !important;
    }

    /* Metrics - bright blue values */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #e8e8e8 !important;
    }

    /* Selectbox and other inputs */
    .stSelectbox label, .stMultiSelect label {
        color: #e8e8e8 !important;
    }
</style>
""", unsafe_allow_html=True)


# Model loading
@st.cache_resource
def load_models():
    """Load all trained models."""
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

    # MAT - Original SOTA model (0.99 AUC)
    mat_path = Path(__file__).parent / 'best_model.pt'
    if mat_path.exists():
        try:
            from app import StereoGNNSmallFinetune
            models['mat'] = StereoGNNSmallFinetune()
            ckpt = torch.load(mat_path, map_location='cpu', weights_only=False)
            models['mat'].load_state_dict(ckpt['model_state_dict'])
            models['mat'].eval()
            print(f"MAT model loaded (AUC: {ckpt.get('best_auc', 'N/A'):.3f})")
        except Exception as e:
            print(f"MAT model loading failed: {e}")
            models['mat'] = None

    models['featurizer'] = MoleculeGraphFeaturizer(use_3d=False)

    return models


def get_fingerprint(smiles: str) -> np.ndarray:
    """Generate Morgan fingerprint + descriptors."""
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


def mol_to_image(smiles: str, size=(400, 300)):
    """Convert SMILES to image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size, fitImage=True)
    return img


def calculate_abuse_liability(dat_probs, net_probs, sert_probs, dat_class,
                               net_class=None, sert_class=None, smiles=None, drug_class=None):
    """Calculate abuse liability score based on DAT activity and drug class.

    Key pharmacological principles:
    1. DAT SUBSTRATES (releasers) >> DAT BLOCKERS >> Inactive
       - Amphetamines (substrates) are highly addictive
       - Cocaine (blocker) is addictive but less than amphetamines
       - Methylphenidate (blocker) has moderate abuse potential

    2. SERT activity REDUCES abuse potential
       - SSRIs have essentially zero abuse potential
       - MDMA's SERT activity moderates its abuse profile

    3. Drug class overrides model predictions for known patterns
    """
    from rdkit import Chem

    # Get active probabilities
    dat_active = dat_probs[1] + dat_probs[2]  # blocker + substrate
    sert_active = sert_probs[1] + sert_probs[2]
    net_active = net_probs[1] + net_probs[2]

    # Drug class detection for rule-based overrides
    detected_class = drug_class
    forced_category = None

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            from rdkit.Chem import Descriptors, rdMolDescriptors
            mw = Descriptors.MolWt(mol)

            # Detect drug classes that have known abuse profiles
            has_phenethylamine = mol.HasSubstructMatch(Chem.MolFromSmarts('NCCc1ccccc1'))
            has_primary_amine = mol.HasSubstructMatch(Chem.MolFromSmarts('[NH2]'))
            has_tropane = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CC2CCC1N2'))
            has_piperidine_phenyl = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNCC1c2ccccc2'))
            has_sulfonamide = mol.HasSubstructMatch(Chem.MolFromSmarts('S(=O)(=O)N'))
            has_fluorine = 'F' in smiles

            # SSRI/SNRI pattern: larger molecule, no primary amine, aryl ether
            aryl_ether = mol.HasSubstructMatch(Chem.MolFromSmarts('cOC'))
            is_ssri_like = (mw > 280 and not has_primary_amine and
                           (aryl_ether or has_fluorine) and sert_class == 'blocker')

            # Amphetamine pattern: small phenethylamine with primary amine
            is_amphetamine_like = (has_phenethylamine and has_primary_amine and
                                   mw < 250 and dat_class == 'substrate')

            # Methylphenidate-like: piperidine + phenyl
            is_methylphenidate_like = has_piperidine_phenyl and mw < 350

            # Cocaine-like: tropane
            is_cocaine_like = has_tropane

            # Apply forced categories based on drug class
            if is_ssri_like or detected_class in ['ssri', 'snri', 'nri']:
                forced_category = "LOW"
                detected_class = detected_class or 'ssri/snri'
            elif detected_class == 'sulfonamide' or has_sulfonamide:
                forced_category = "LOW"
            elif detected_class in ['gabapentinoid']:
                forced_category = "LOW"
            elif is_amphetamine_like or detected_class in ['amphetamine', 'cathinone']:
                forced_category = "HIGH"
            elif is_cocaine_like:
                forced_category = "HIGH"
            elif is_methylphenidate_like:
                forced_category = "MODERATE"
                detected_class = 'methylphenidate-like'

    # If we have a forced category from drug class, return it
    if forced_category:
        if forced_category == "HIGH":
            score = 85
        elif forced_category == "MODERATE":
            score = 50
        else:
            score = 20
        return score, forced_category

    # Otherwise, calculate based on MAT activity

    # REVISED SCORING: Substrates >> Blockers >> Inactive
    if dat_class == 'substrate':
        # Releasers (amphetamine-like) - HIGHEST risk
        # Score: 70-95 depending on confidence
        base = 70 + (dat_probs[2] * 25)
    elif dat_class == 'blocker':
        # Blockers (cocaine-like) - MODERATE-HIGH risk
        # Key insight: blockers are LESS abusable than substrates
        # Score: 40-65 depending on confidence
        base = 40 + (dat_probs[1] * 25)
    else:
        # Inactive at DAT - LOW risk
        # Score: 5-25
        base = 5 + (dat_active * 20)

    # SERT activity REDUCES abuse potential (key pharmacological insight)
    # SSRIs are not abused despite CNS activity
    if sert_class == 'blocker' and dat_class != 'substrate':
        # SERT blocker without DAT substrate = reduced abuse
        sert_reduction = 15
        base = max(5, base - sert_reduction)
    elif sert_class == 'substrate':
        # SERT substrate (MDMA-like) - moderate reduction
        sert_reduction = 5
        base = max(5, base - sert_reduction)

    # DAT selectivity bonus - only applies if DAT active
    if dat_class in ['blocker', 'substrate'] and sert_active < 0.3:
        # High DAT, low SERT = more selective = more abuse potential
        selectivity_bonus = 10
        base += selectivity_bonus

    score = min(100, max(0, base))

    # Thresholds adjusted for better calibration
    if score >= 65:
        category = "HIGH"
    elif score >= 40:
        category = "MODERATE"
    else:
        category = "LOW"

    return score, category


@torch.no_grad()
def predict_molecule(smiles: str, models: dict) -> dict:
    """Run all predictions on a molecule."""
    results = {'smiles': smiles, 'valid': False}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        results['error'] = "Invalid SMILES"
        return results

    results['valid'] = True
    results['mol_weight'] = Descriptors.MolWt(mol)
    results['logp'] = Descriptors.MolLogP(mol)
    results['tpsa'] = Descriptors.TPSA(mol)
    results['hbd'] = Descriptors.NumHDonors(mol)
    results['hba'] = Descriptors.NumHAcceptors(mol)

    # Featurize
    try:
        data = models['featurizer'].featurize(smiles, {'DAT': -1, 'NET': -1, 'SERT': -1})
        if data is None:
            results['error'] = "Featurization failed"
            return results

        fp = get_fingerprint(smiles)
        if fp is not None:
            data.fp = torch.tensor(fp, dtype=torch.float32).unsqueeze(0)

        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data])

        # hERG prediction
        if 'herg' in models:
            output = models['herg'](batch)
            prob = torch.softmax(output['logits'], dim=-1)[0, 1].item()
            results['herg_prob'] = prob
            results['herg_risk'] = 'HIGH' if prob > 0.7 else ('MODERATE' if prob > 0.3 else 'LOW')

        # CYP prediction
        if 'cyp' in models:
            # Add mask and y for CYP model
            n_targets = len(models['cyp_targets'])
            batch.mask = torch.ones(1, n_targets)
            batch.y = torch.zeros(1, n_targets, dtype=torch.long)

            output = models['cyp'](batch)
            results['cyp'] = {}
            for i, cyp in enumerate(models['cyp_targets']):
                prob = torch.softmax(output[cyp], dim=-1)[0, 1].item()
                results['cyp'][cyp] = {
                    'prob': prob,
                    'inhibitor': prob > 0.5
                }

        # MAT prediction (original SOTA model + rule corrections)
        if 'mat' in models and models['mat'] is not None:
            mat_output = models['mat'](batch)

            # Get raw predictions first
            raw_preds = {}
            for target in ['DAT', 'NET', 'SERT']:
                logits = mat_output[target][0]
                probs = torch.softmax(logits, dim=-1)
                max_prob = probs.max().item()
                pred_class = ['inactive', 'blocker', 'substrate'][probs.argmax().item()]
                raw_preds[target] = {
                    'class': pred_class,
                    'probs': probs.numpy(),
                    'max_prob': max_prob,
                }

            # Apply pharmacology rules for correction
            corrected_preds = apply_pharmacology_rules(smiles, raw_preds)

            results['mat'] = {}
            for target in ['DAT', 'NET', 'SERT']:
                pred_class = corrected_preds[target]['class']
                max_prob = corrected_preds[target]['max_prob']
                was_corrected = corrected_preds[target].get('corrected', False)

                # Confidence assessment
                if was_corrected:
                    confidence = 'rule-based'
                elif max_prob >= 0.8:
                    confidence = 'high'
                elif max_prob >= 0.6:
                    confidence = 'medium'
                else:
                    confidence = 'low'

                results['mat'][target] = {
                    'class': pred_class,
                    'probs': corrected_preds[target]['probs'],
                    'confidence': confidence,
                    'max_prob': max_prob,
                    'corrected': was_corrected,
                }

            # Store drug class if detected
            if corrected_preds.get('drug_class'):
                results['drug_class'] = corrected_preds['drug_class']

            # Abuse liability - use dedicated AbusePredictor
            from abuse_predictor import AbusePredictor
            abuse_predictor = AbusePredictor()

            dat_probs = results['mat']['DAT']['probs']
            net_probs = results['mat']['NET']['probs']
            sert_probs = results['mat']['SERT']['probs']
            dat_class = results['mat']['DAT']['class']
            net_class = results['mat']['NET']['class']
            sert_class = results['mat']['SERT']['class']

            score, category = abuse_predictor.predict(
                smiles=smiles,
                dat_class=dat_class,
                net_class=net_class,
                sert_class=sert_class,
                dat_probs=np.array(dat_probs),
                net_probs=np.array(net_probs),
                sert_probs=np.array(sert_probs),
            )
            results['abuse_score'] = score
            results['abuse_category'] = category

    except Exception as e:
        results['error'] = str(e)

    return results


def render_metric_card(label: str, value: str, sublabel: str = None):
    """Render a styled metric card."""
    sublabel_html = f'<div style="color:#666;font-size:0.8rem;">{sublabel}</div>' if sublabel else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sublabel_html}
    </div>
    """, unsafe_allow_html=True)


def render_badge(text: str, level: str):
    """Render a status badge."""
    badge_class = {
        'LOW': 'badge-safe',
        'MODERATE': 'badge-warning',
        'HIGH': 'badge-danger',
        'SAFE': 'badge-safe',
        'RISK': 'badge-danger'
    }.get(level, 'badge-warning')

    return f'<span class="{badge_class}">{text}</span>'


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 40px 0;">
        <h1>🧬 Insilico Drug Discovery Toolkit</h1>
        <p style="color: #888; font-size: 1.2rem;">
            SOTA Molecular Property Prediction • Stereo-Aware GNN • Multi-Task Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        | Module | AUC |
        |--------|-----|
        | hERG Cardiotox | **0.91** |
        | CYP Mean | **0.88** |
        | MAT Kinetics | **0.90** |
        """)

        st.markdown("---")
        st.markdown("### 📊 Modules")
        st.markdown("""
        - **hERG** - Cardiac toxicity risk
        - **CYP450** - Drug metabolism
        - **MAT** - Transporter kinetics
        - **Abuse** - Liability scoring
        """)

        st.markdown("---")
        with st.expander("About Dis-Solved"):
            st.markdown("""
**Dis-Solved** is an independent AI-powered platform dedicated to making advanced in silico drug discovery accessible, affordable, and targeted—especially for central nervous system (CNS) therapeutics and safer stimulant design.

Founded and built single-handedly by **Nabil Yasini**, Dis-Solved started as a personal project on a modest laptop: developing high-performance, interpretable machine learning models that outperform many state-of-the-art benchmarks in key ADMET predictions. What began as a breakthrough blood-brain barrier permeability (BBBP) predictor (achieving 0.92 internal AUC and 0.96 on external validation, with explicit stereoisomer training) has evolved into a modular toolkit tailored for early-stage screening of ADHD and neuropsychiatric drug candidates.

**Our core focus:**
- Specialized predictions for monoamine transporter activity (including kinetics for DAT, NET, SERT)
- Toxicology
- PK/PD curve simulation
- Abuse potential scoring—to help design lower-liability stimulants

By leveraging clever data augmentation and modern graph neural networks, we've created tools that are not only accurate and explainable but also beginner-friendly and priced for real-world use—starting from free tiers up to affordable subscriptions.

At Dis-Solved, we believe cutting-edge AI shouldn't be locked behind enterprise paywalls or massive teams. We're here to empower academics, small biotechs, indie researchers, and drug hunters to screen smarter, fail faster, and advance safer CNS therapies.

*Whether you're a student exploring computational chemistry or a startup optimizing leads, Dis-Solved is your practical partner in dissolving the toughest challenges in drug discovery.*
            """)

        st.markdown("""
        <div style="color:#666; font-size:0.8rem;">
        Built with StereoGNN<br>
        Fingerprint Fusion Architecture<br>
        © 2024 Dis-Solved
        </div>
        """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        models = load_models()

    # Main input
    col1, col2 = st.columns([3, 1])

    with col1:
        smiles_input = st.text_input(
            "Enter SMILES",
            placeholder="e.g., CC(N)Cc1ccccc1 (Amphetamine)",
            label_visibility="collapsed"
        )

    with col2:
        predict_btn = st.button("🔬 Analyze", use_container_width=True)

    # Example molecules
    st.markdown("**Quick Examples:**")
    examples = {
        "Amphetamine": "CC(N)Cc1ccccc1",
        "Cocaine": "COC(=O)C1C2CCC(CC1OC(=O)c1ccccc1)N2C",
        "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "Methylphenidate": "COC(=O)C(c1ccccc1)C1CCCCN1",
        "Verapamil": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC"
    }

    cols = st.columns(len(examples))
    for i, (name, smi) in enumerate(examples.items()):
        if cols[i].button(name, key=f"ex_{name}"):
            smiles_input = smi
            predict_btn = True

    # Run prediction
    if predict_btn and smiles_input:
        # Resolve drug name to SMILES if needed
        with st.spinner("Looking up compound..."):
            smiles, was_resolved, original_name = resolve_input(smiles_input)

        if was_resolved:
            st.success(f"🔍 Found **{original_name}** → `{smiles}`")

        results = predict_molecule(smiles, models)

        if not results['valid']:
            st.error(f"❌ {results.get('error', 'Invalid molecule')}")
            return

        st.markdown("---")

        # Molecule visualization
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 🔬 Structure")
            img = mol_to_image(smiles)
            if img:
                st.image(img, use_container_width=True)

            st.markdown("### 📋 Properties")
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | MW | {results['mol_weight']:.1f} |
            | LogP | {results['logp']:.2f} |
            | TPSA | {results['tpsa']:.1f} |
            | HBD | {results['hbd']} |
            | HBA | {results['hba']} |
            """)

        with col2:
            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["❤️ Cardiac", "💊 Metabolism", "🧠 Transporters", "⚠️ Abuse"])

            with tab1:
                st.markdown("### hERG Cardiotoxicity")
                if 'herg_prob' in results:
                    prob = results['herg_prob']
                    risk = results['herg_risk']

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Block Probability", f"{prob:.1%}")
                    with col_b:
                        st.markdown(f"**Risk Level:** {render_badge(risk, risk)}", unsafe_allow_html=True)

                    st.progress(prob)

                    if risk == 'HIGH':
                        st.warning("⚠️ High risk of QT prolongation. Consider structural modifications.")
                    elif risk == 'MODERATE':
                        st.info("ℹ️ Moderate hERG activity. Further testing recommended.")
                    else:
                        st.success("✅ Low cardiac liability predicted.")
                else:
                    st.info("hERG model not loaded")

            with tab2:
                st.markdown("### CYP450 Inhibition")
                if 'cyp' in results:
                    cyp_data = []
                    for cyp, data in results['cyp'].items():
                        cyp_data.append({
                            'Enzyme': cyp,
                            'Probability': f"{data['prob']:.1%}",
                            'Status': '🔴 Inhibitor' if data['inhibitor'] else '🟢 Non-inhibitor'
                        })

                    df = pd.DataFrame(cyp_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    inhibitors = [cyp for cyp, d in results['cyp'].items() if d['inhibitor']]
                    if inhibitors:
                        st.warning(f"⚠️ Predicted CYP inhibitor: {', '.join(inhibitors)}")
                    else:
                        st.success("✅ No significant CYP inhibition predicted.")
                else:
                    st.info("CYP model not loaded")

            with tab3:
                st.markdown("### Monoamine Transporters")
                if 'mat' in results:
                    for target in ['DAT', 'NET', 'SERT']:
                        data = results['mat'][target]
                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.markdown(f"**{target}**")
                        with col_b:
                            class_badge = {
                                'substrate': '🟢 Substrate',
                                'blocker': '🟡 Blocker',
                                'inactive': '⚪ Inactive'
                            }.get(data['class'], data['class'])
                            st.markdown(class_badge)
                        with col_c:
                            # Confidence indicator
                            conf = data.get('confidence', 'unknown')
                            conf_color = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(conf, '⚪')
                            prob_pct = data.get('max_prob', 0) * 100
                            st.markdown(f"{conf_color} {prob_pct:.0f}% conf")

                    # Low confidence warning
                    low_conf_targets = [t for t in ['DAT', 'NET', 'SERT']
                                        if results['mat'][t].get('confidence') == 'low']
                    if low_conf_targets:
                        st.warning(f"⚠️ Low confidence for: {', '.join(low_conf_targets)}. "
                                   "Predictions may be unreliable for novel structures.")

                    st.markdown("---")
                    st.markdown("**Selectivity Profile**")

                    probs = {t: results['mat'][t]['probs'][2] for t in ['DAT', 'NET', 'SERT']}
                    chart_data = pd.DataFrame({
                        'Transporter': list(probs.keys()),
                        'Substrate Probability': list(probs.values())
                    })
                    st.bar_chart(chart_data.set_index('Transporter'))
                else:
                    st.info("MAT model not loaded")

            with tab4:
                st.markdown("### Abuse Liability Assessment")
                if 'abuse_score' in results:
                    score = results['abuse_score']
                    category = results['abuse_category']

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Abuse Score", f"{score:.0f}/100")
                    with col_b:
                        st.markdown(f"**Category:** {render_badge(category, category)}", unsafe_allow_html=True)

                    st.progress(score / 100)

                    st.markdown("""
                    **Scoring Factors:**
                    - DAT activity (primary driver)
                    - DAT/NET/SERT selectivity
                    - Substrate vs blocker mechanism
                    """)

                    if category == 'HIGH':
                        st.error("🚨 High abuse potential. Consider prodrug strategies or formulation controls.")
                    elif category == 'MODERATE':
                        st.warning("⚠️ Moderate abuse potential. Schedule considerations may apply.")
                    else:
                        st.success("✅ Low abuse liability predicted.")
                else:
                    st.info("Requires MAT predictions")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Insilico Drug Discovery Toolkit • Stereo-Aware GNN Architecture • Research Use Only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
