"""
Insilico Drug Discovery Toolkit - BETA
=======================================
Descriptor-based ADMET predictions.
Full GNN version available for purchase.

Run: streamlit run app_ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, DataStructs
from rdkit.Chem import rdMolDescriptors
import requests
import urllib.parse
import io
import base64

st.set_page_config(
    page_title="Insilico Drug Discovery Toolkit | BETA",
    page_icon="🧬",
    layout="wide"
)

# Cache for PubChem lookups
_smiles_cache = {}


def name_to_smiles(name: str) -> str:
    """Convert any chemical/drug name to SMILES using PubChem API with synonym search."""
    if not name:
        return None

    name = name.strip()
    cache_key = name.lower()

    if cache_key in _smiles_cache:
        return _smiles_cache[cache_key]

    # Check if already valid SMILES
    mol = Chem.MolFromSmiles(name)
    if mol is not None:
        return name

    encoded_name = urllib.parse.quote(name)

    # Method 1: Direct name lookup
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IsomericSMILES,CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            smiles = props.get('IsomericSMILES') or props.get('CanonicalSMILES')
            if smiles:
                _smiles_cache[cache_key] = smiles
                return smiles
    except:
        pass

    # Method 2: Search by synonym (finds more names)
    try:
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            if cids:
                cid = cids[0]
                prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
                prop_resp = requests.get(prop_url, timeout=10)
                if prop_resp.status_code == 200:
                    prop_data = prop_resp.json()
                    props = prop_data.get('PropertyTable', {}).get('Properties', [{}])[0]
                    smiles = props.get('IsomericSMILES')
                    if smiles:
                        _smiles_cache[cache_key] = smiles
                        return smiles
    except:
        pass

    return None


def mol_to_image(smiles: str, size=(350, 250)):
    """Convert SMILES to image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size, fitImage=True)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_descriptors(smiles: str) -> dict:
    """Calculate molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'fsp3': Descriptors.FractionCSP3(mol),
        'formula': rdMolDescriptors.CalcMolFormula(mol),
    }


def predict_herg(smiles: str, desc: dict) -> dict:
    """Predict hERG cardiotoxicity risk using descriptors."""
    mol = Chem.MolFromSmiles(smiles)

    # hERG blockers tend to have: high logP, basic nitrogen, aromatic rings
    score = 0.3  # baseline

    if desc['logp'] > 3.5:
        score += 0.2
    if desc['logp'] > 4.5:
        score += 0.15

    if desc['mw'] > 400:
        score += 0.1

    # Basic nitrogen (common in hERG blockers)
    basic_n = mol.HasSubstructMatch(Chem.MolFromSmarts('[#7;+]')) or \
              mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]'))
    if basic_n:
        score += 0.15

    if desc['aromatic_rings'] >= 2:
        score += 0.1

    # Piperidine/piperazine (common in hERG blockers)
    if mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNCC1')) or \
       mol.HasSubstructMatch(Chem.MolFromSmarts('C1CNCCN1')):
        score += 0.1

    score = min(0.95, max(0.05, score))

    risk = "HIGH" if score > 0.6 else "MODERATE" if score > 0.4 else "LOW"

    return {'probability': score, 'risk': risk}


def predict_cyp_inhibition(smiles: str, desc: dict) -> dict:
    """Predict CYP450 inhibition using descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    results = {}

    cyp_targets = ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4']

    for cyp in cyp_targets:
        score = 0.25  # baseline

        # General CYP inhibitor features
        if desc['logp'] > 3:
            score += 0.1
        if desc['aromatic_rings'] >= 2:
            score += 0.1
        if desc['mw'] > 300:
            score += 0.05

        # CYP-specific patterns
        if cyp == 'CYP1A2':
            # Planar aromatics
            if desc['aromatic_rings'] >= 3:
                score += 0.15
        elif cyp == 'CYP2D6':
            # Basic nitrogen
            if mol.HasSubstructMatch(Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]')):
                score += 0.15
        elif cyp == 'CYP3A4':
            # Large molecules
            if desc['mw'] > 500:
                score += 0.15

        score = min(0.9, max(0.1, score))
        risk = "HIGH" if score > 0.6 else "MODERATE" if score > 0.4 else "LOW"
        results[cyp] = {'probability': score, 'risk': risk}

    return results


def predict_mat_activity(smiles: str, desc: dict) -> dict:
    """Predict monoamine transporter activity using structural patterns."""
    mol = Chem.MolFromSmiles(smiles)
    results = {}

    # Phenethylamine core (common in MAT actives)
    has_phenethylamine = mol.HasSubstructMatch(Chem.MolFromSmarts('NCCc1ccccc1'))
    has_amphetamine = mol.HasSubstructMatch(Chem.MolFromSmarts('NC(C)Cc1ccccc1'))
    has_cathinone = mol.HasSubstructMatch(Chem.MolFromSmarts('NC(C)C(=O)c1ccccc1'))
    has_tropane = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CC2CCC1N2'))
    has_piperidine = mol.HasSubstructMatch(Chem.MolFromSmarts('C1CCNCC1'))

    for target in ['DAT', 'NET', 'SERT']:
        probs = {'substrate': 0.1, 'blocker': 0.15, 'inactive': 0.75}

        if has_phenethylamine or has_amphetamine:
            if target == 'DAT':
                probs = {'substrate': 0.7, 'blocker': 0.2, 'inactive': 0.1}
            elif target == 'NET':
                probs = {'substrate': 0.6, 'blocker': 0.25, 'inactive': 0.15}
            elif target == 'SERT':
                probs = {'substrate': 0.3, 'blocker': 0.3, 'inactive': 0.4}

        if has_cathinone:
            probs = {'substrate': 0.65, 'blocker': 0.2, 'inactive': 0.15}

        if has_tropane:  # Cocaine-like
            probs = {'substrate': 0.1, 'blocker': 0.75, 'inactive': 0.15}

        # SSRI-like pattern (diphenyl with amine)
        if mol.HasSubstructMatch(Chem.MolFromSmarts('c1ccccc1Oc2ccccc2')) and \
           mol.HasSubstructMatch(Chem.MolFromSmarts('CNCC')):
            if target == 'SERT':
                probs = {'substrate': 0.05, 'blocker': 0.8, 'inactive': 0.15}
            else:
                probs = {'substrate': 0.05, 'blocker': 0.2, 'inactive': 0.75}

        pred_class = max(probs, key=probs.get)
        results[target] = {'class': pred_class, 'probabilities': probs}

    return results


def predict_abuse_liability(mat_results: dict, smiles: str) -> dict:
    """Predict abuse liability based on MAT activity."""
    mol = Chem.MolFromSmiles(smiles)

    dat = mat_results['DAT']

    # High abuse: DAT substrates (releasers like amphetamines)
    if dat['class'] == 'substrate' and dat['probabilities']['substrate'] > 0.5:
        return {'level': 'HIGH', 'score': 0.85, 'reason': 'DAT substrate (releaser) - high addiction potential'}

    # Cocaine-like blockers
    if dat['class'] == 'blocker' and dat['probabilities']['blocker'] > 0.6:
        if mol.HasSubstructMatch(Chem.MolFromSmarts('C1CC2CCC1N2')):  # tropane
            return {'level': 'HIGH', 'score': 0.8, 'reason': 'Cocaine-like DAT blocker'}
        return {'level': 'MODERATE', 'score': 0.5, 'reason': 'DAT blocker - moderate abuse potential'}

    # SERT-selective (SSRIs) - low abuse
    sert = mat_results['SERT']
    if sert['class'] == 'blocker' and dat['class'] == 'inactive':
        return {'level': 'LOW', 'score': 0.1, 'reason': 'SERT-selective - minimal abuse potential'}

    return {'level': 'LOW', 'score': 0.2, 'reason': 'No significant DAT activity'}


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; margin-bottom: 0.5rem;">
            🧬 Insilico Drug Discovery Toolkit
        </h1>
        <p style="color: #ffa500; font-weight: bold; font-size: 1.2rem;">BETA VERSION</p>
        <p style="color: #888;">Descriptor-based ADMET predictions | Full GNN models available for licensing</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        **BETA Version**

        This demo uses descriptor-based predictions.

        **Full Version Features:**
        - Graph Neural Network models
        - 0.968 AUC on MAT prediction
        - 0.91 AUC on hERG prediction
        - Batch processing
        - API access

        Contact: nabilyasini@example.com
        """)

        st.markdown("---")
        st.markdown("**Predictions Available:**")
        st.markdown("- MAT Activity (DAT/NET/SERT)")
        st.markdown("- Abuse Liability")
        st.markdown("- hERG Cardiotoxicity")
        st.markdown("- CYP450 Inhibition")

    # Main input
    st.subheader("Enter Molecule")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Enter SMILES or drug name",
            placeholder="e.g., Amphetamine, Cocaine, Fluoxetine, or paste SMILES",
            label_visibility="collapsed"
        )
    with col2:
        predict_btn = st.button("Analyze", type="primary", use_container_width=True)

    # Quick examples
    st.markdown("**Examples:**")
    examples = ["Amphetamine", "Cocaine", "Fluoxetine", "Caffeine", "Morphine", "Diazepam"]
    cols = st.columns(6)
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state['input'] = ex
                st.rerun()

    if 'input' in st.session_state:
        user_input = st.session_state['input']
        del st.session_state['input']
        predict_btn = True

    if predict_btn and user_input:
        # Resolve input
        with st.spinner("Looking up molecule..."):
            smiles = name_to_smiles(user_input)

        if not smiles:
            st.error(f"Could not resolve '{user_input}'. Please enter a valid SMILES or drug name.")
            return

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid molecule structure.")
            return

        st.success(f"**{user_input.title()}**: `{smiles}`")

        # Calculate descriptors
        desc = get_descriptors(smiles)

        # Display molecule and properties
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            img_b64 = mol_to_image(smiles)
            if img_b64:
                st.image(f"data:image/png;base64,{img_b64}", caption=user_input.title())

            st.markdown("**Properties:**")
            st.markdown(f"- Formula: {desc['formula']}")
            st.markdown(f"- MW: {desc['mw']:.1f} Da")
            st.markdown(f"- LogP: {desc['logp']:.2f}")
            st.markdown(f"- TPSA: {desc['tpsa']:.1f} Å²")
            st.markdown(f"- HBD/HBA: {desc['hbd']}/{desc['hba']}")

        with col2:
            # Predictions
            tabs = st.tabs(["MAT Activity", "Abuse Liability", "hERG", "CYP450"])

            with tabs[0]:
                st.markdown("### Monoamine Transporter Activity")
                mat_results = predict_mat_activity(smiles, desc)

                for target in ['DAT', 'NET', 'SERT']:
                    r = mat_results[target]
                    color = {"substrate": "🟢", "blocker": "🟡", "inactive": "⚪"}[r['class']]
                    st.markdown(f"**{target}**: {color} {r['class'].upper()}")

                    prob_str = " | ".join([f"{k}: {v:.0%}" for k, v in r['probabilities'].items()])
                    st.caption(prob_str)

            with tabs[1]:
                st.markdown("### Abuse Liability Assessment")
                abuse = predict_abuse_liability(mat_results, smiles)

                color_map = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}
                st.markdown(f"## {color_map[abuse['level']]} {abuse['level']}")
                st.markdown(f"**Score:** {abuse['score']:.2f}")
                st.markdown(f"**Rationale:** {abuse['reason']}")

            with tabs[2]:
                st.markdown("### hERG Cardiotoxicity")
                herg = predict_herg(smiles, desc)

                color_map = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}
                st.markdown(f"## {color_map[herg['risk']]} {herg['risk']} Risk")
                st.markdown(f"**Probability:** {herg['probability']:.2f}")

                st.caption("hERG channel blocking can cause cardiac arrhythmias (QT prolongation)")

            with tabs[3]:
                st.markdown("### CYP450 Inhibition")
                cyp = predict_cyp_inhibition(smiles, desc)

                for enzyme, r in cyp.items():
                    color = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[r['risk']]
                    st.markdown(f"**{enzyme}**: {color} {r['risk']} ({r['probability']:.0%})")

        # Disclaimer
        st.markdown("---")
        st.warning("""
        **BETA Version Disclaimer:** These predictions use descriptor-based models for demonstration.
        For research or commercial use, please contact us for the full GNN-powered version with validated performance metrics.
        """)


if __name__ == "__main__":
    main()
