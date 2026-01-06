"""
Pull additional hERG data from multiple sources.
Merge with ChEMBL clean data for maximum coverage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
import requests
from io import StringIO
from chembl_webresource_client.new_client import new_client

# CiPA cardiac panel
HERG_CHEMBL_ID = 'CHEMBL240'    # hERG - repolarization
NAV15_CHEMBL_ID = 'CHEMBL1845'  # Nav1.5 - depolarization
CAV12_CHEMBL_ID = 'CHEMBL1862'  # Cav1.2 - plateau/contraction

def get_cardiac_panel():
    """Pull full CiPA cardiac ion channel panel from ChEMBL."""
    print("Fetching CiPA cardiac panel (Nav1.5 + Cav1.2)...")
    activity = new_client.activity
    all_data = []

    for name, chembl_id in [('Nav1.5', NAV15_CHEMBL_ID), ('Cav1.2', CAV12_CHEMBL_ID)]:
        print(f"  Fetching {name}...")
        try:
            results = activity.filter(
                target_chembl_id=chembl_id,
                standard_type__in=['IC50', 'Ki'],
                standard_units='nM',
                standard_relation='=',
            ).only(['canonical_smiles', 'standard_value', 'pchembl_value'])

            records = []
            for r in results:
                if r.get('canonical_smiles') and r.get('standard_value'):
                    val = float(r['standard_value'])
                    pval = float(r['pchembl_value']) if r.get('pchembl_value') else -np.log10(val * 1e-9)
                    records.append({
                        'smiles': r['canonical_smiles'],
                        'pIC50': pval,
                        'active': int(pval > 5),  # IC50 < 10µM
                        'source': f'ChEMBL_{name}',
                    })
            df = pd.DataFrame(records)
            df = df.drop_duplicates('smiles', keep='first')
            all_data.append(df)
            print(f"    {name}: {len(df)} molecules")
        except Exception as e:
            print(f"    {name} failed: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def get_tdc_herg():
    """Pull hERG + QT data from Therapeutics Data Commons."""
    print("Fetching TDC hERG + QT datasets...")
    all_dfs = []

    try:
        from tdc.single_pred import Tox
        # hERG inhibition
        data = Tox(name='hERG')
        df = data.get_data()
        df = df.rename(columns={'Drug': 'smiles', 'Y': 'active'})
        df['source'] = 'TDC_hERG'
        df['pIC50'] = np.nan
        all_dfs.append(df[['smiles', 'active', 'pIC50', 'source']])
        print(f"  TDC hERG: {len(df)} molecules")
    except Exception as e:
        print(f"  TDC hERG failed: {e}")

    try:
        from tdc.single_pred import Tox
        # QT prolongation (direct cardiac measurement)
        data = Tox(name='QT')
        df = data.get_data()
        df = df.rename(columns={'Drug': 'smiles', 'Y': 'active'})
        df['source'] = 'TDC_QT'
        df['pIC50'] = np.nan
        all_dfs.append(df[['smiles', 'active', 'pIC50', 'source']])
        print(f"  TDC QT: {len(df)} molecules")
    except Exception as e:
        print(f"  TDC QT failed: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def get_pubchem_herg():
    """Pull hERG data from PubChem BioAssay - including cardiac safety assays."""
    print("Fetching PubChem hERG + cardiac bioassays...")

    # Key hERG and cardiac safety assays in PubChem
    assay_ids = [
        376,      # hERG patch clamp
        588834,   # hERG inhibition
        493208,   # hERG blockade
        1259405,  # hERG fluorescence
        1511882,  # Cardiac ion channel panel
        2289,     # hERG binding
        652054,   # hERG automated patch clamp
    ]

    all_data = []
    for aid in assay_ids:
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                if 'PUBCHEM_EXT_DATASOURCE_SMILES' in df.columns:
                    df = df.rename(columns={
                        'PUBCHEM_EXT_DATASOURCE_SMILES': 'smiles',
                        'PUBCHEM_ACTIVITY_OUTCOME': 'outcome'
                    })
                    df['active'] = (df['outcome'] == 'Active').astype(int)
                    df['source'] = f'PubChem_{aid}'
                    df['pIC50'] = np.nan
                    all_data.append(df[['smiles', 'active', 'pIC50', 'source']])
                    print(f"  AID {aid}: {len(df)} records")
        except Exception as e:
            print(f"  AID {aid} failed: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def validate_smiles(smiles):
    """Check if SMILES is valid."""
    if not smiles or pd.isna(smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False

def merge_all():
    """Merge all hERG data sources."""
    data_dir = Path('data/toxicity')

    # Load existing clean ChEMBL data
    chembl_path = data_dir / 'hERG_clean.csv'
    if chembl_path.exists():
        chembl = pd.read_csv(chembl_path)
        chembl['source'] = 'ChEMBL'
        print(f"ChEMBL clean: {len(chembl)} molecules")
    else:
        print("No ChEMBL clean data found!")
        chembl = pd.DataFrame()

    # Get additional sources
    tdc = get_tdc_herg()
    pubchem = get_pubchem_herg()
    cardiac = get_cardiac_panel()

    # Merge all
    all_data = []
    if len(chembl) > 0:
        all_data.append(chembl[['smiles', 'active', 'pIC50', 'source']])
    if len(tdc) > 0:
        all_data.append(tdc)
    if len(pubchem) > 0:
        all_data.append(pubchem)
    if len(cardiac) > 0:
        all_data.append(cardiac)

    if not all_data:
        print("No data collected!")
        return

    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal raw: {len(df)}")

    # Validate SMILES
    df = df[df['smiles'].apply(validate_smiles)]
    print(f"Valid SMILES: {len(df)}")

    # Canonicalize
    df['smiles'] = df['smiles'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))

    # Deduplicate - prefer ChEMBL (has pIC50), then by activity
    df['has_pic50'] = df['pIC50'].notna()
    df = df.sort_values(['has_pic50', 'active'], ascending=[False, False])
    df = df.drop_duplicates('smiles', keep='first')
    df = df.drop('has_pic50', axis=1)

    # Add stereo flag
    df['has_stereo'] = df['smiles'].str.contains(r'@|\\|/', regex=True)

    print(f"\nFinal merged dataset:")
    print(f"  Unique molecules: {len(df)}")
    print(f"  With pIC50: {df['pIC50'].notna().sum()}")
    print(f"  Active: {(df['active'] == 1).sum()}")
    print(f"  Inactive: {(df['active'] == 0).sum()}")
    print(f"  With stereo: {df['has_stereo'].sum()}")
    print(f"\nBy source:")
    print(df['source'].value_counts())

    # Save
    out_path = data_dir / 'hERG_merged.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    merge_all()
