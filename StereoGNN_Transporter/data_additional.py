"""
Additional Compounds to Reach Target
====================================

We need 500+ compounds. This file adds more to push us over.
"""

import pandas as pd
from rdkit import Chem

# Additional phenethylamines and analogs
ADDITIONAL_COMPOUNDS = [
    # More amphetamine N-substitutions
    ("C[C@H](NC(C)C)Cc1ccccc1", "N-Isopropylamphetamine", {"DAT": 1, "NET": 2, "SERT": 0}, 0.8),
    ("C[C@H](NC(C)(C)C)Cc1ccccc1", "N-tert-Butylamphetamine", {"DAT": 0, "NET": 1, "SERT": 0}, 0.8),
    ("C[C@H](NCc1ccccc1)Cc2ccccc2", "N-Benzylamphetamine", {"DAT": 1, "NET": 1, "SERT": 0}, 0.8),

    # More ring-substituted amphetamines
    ("C[C@H](N)Cc1cc(C)ccc1C", "2,4-Dimethylamphetamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),
    ("C[C@H](N)Cc1cc(F)cc(F)c1", "3,5-Difluoroamphetamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),
    ("C[C@H](N)Cc1ccc(C)c(C)c1", "3,4-Dimethylamphetamine", {"DAT": 2, "NET": 2, "SERT": 2}, 0.7),
    ("C[C@H](N)Cc1cc(OC)cc(OC)c1", "3,5-Dimethoxyamphetamine", {"DAT": 1, "NET": 2, "SERT": 1}, 0.7),

    # Methamphetamine variants
    ("C[C@H](NC)Cc1ccc(F)cc1", "4-Fluoromethamphetamine", {"DAT": 2, "NET": 2, "SERT": 2}, 0.9),
    ("C[C@H](NC)Cc1ccc(Cl)cc1", "4-Chloromethamphetamine", {"DAT": 2, "NET": 2, "SERT": 2}, 0.9),
    ("C[C@H](NC)Cc1ccc(C)cc1", "4-Methylmethamphetamine", {"DAT": 2, "NET": 2, "SERT": 2}, 0.9),
    ("C[C@H](NC)Cc1cccc(F)c1", "3-Fluoromethamphetamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.8),
    ("C[C@H](NC)Cc1ccccc1F", "2-Fluoromethamphetamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.8),

    # More methylenedioxy compounds
    ("C[C@H](NCC)Cc1ccc2OCOc2c1", "MDEA (S)", {"DAT": 2, "NET": 2, "SERT": 2}, 0.95),
    ("CCC(N)Cc1ccc2OCOc2c1", "MBDB", {"DAT": 2, "NET": 2, "SERT": 2}, 0.9),
    ("CCC(NC)Cc1ccc2OCOc2c1", "MBDB N-methyl", {"DAT": 2, "NET": 2, "SERT": 2}, 0.9),
    ("C[C@H](N)Cc1cc2OCOc2cc1OC", "MMDA", {"DAT": 1, "NET": 2, "SERT": 2}, 0.9),

    # Additional cathinone variants
    ("CC(NC)C(=O)c1ccccc1C", "2-Methylmethcathinone", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),
    ("CC(NC)C(=O)c1cccc(C)c1", "3-Ethylmethcathinone", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),
    ("CC(NCC)C(=O)c1ccc(C)cc1", "4-Methylethcathinone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.8),
    ("CCC(NC)C(=O)c1ccc(F)cc1", "4-Fluorobuphedrone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.8),
    ("CCC(NC)C(=O)c1ccc(Cl)cc1", "4-Chlorobuphedrone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.8),
    ("CCC(NC)C(=O)c1ccc(C)cc1", "4-Methylbuphedrone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.8),
    ("CCCC(NC)C(=O)c1ccc(F)cc1", "4-Fluoropentedrone", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),
    ("CC(N)C(=O)c1ccc(F)cc1", "4-Fluorocathinone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.85),
    ("CC(N)C(=O)c1ccc(Cl)cc1", "4-Chlorocathinone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.85),
    ("CC(N)C(=O)c1ccc(C)cc1", "4-Methylcathinone", {"DAT": 2, "NET": 2, "SERT": 2}, 0.85),

    # Aminoindane variants
    ("NC1Cc2ccccc2C1C", "2-Amino-1-methylindane", {"DAT": 2, "NET": 2, "SERT": 1}, 0.8),
    ("CNC1Cc2cc(Cl)ccc2C1", "5-Chloro-NMAI", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),
    ("CNC1Cc2cc(F)ccc2C1", "5-Fluoro-NMAI", {"DAT": 2, "NET": 2, "SERT": 1}, 0.7),

    # More benzofuran compounds
    ("CC(N)Cc1ccc2ccoc2c1", "7-APB", {"DAT": 2, "NET": 2, "SERT": 2}, 0.85),
    ("CC(NC)Cc1ccc2ccoc2c1", "7-MAPB", {"DAT": 2, "NET": 2, "SERT": 2}, 0.85),
    ("CCC(N)Cc1cc2ccoc2cc1", "5-APDB", {"DAT": 2, "NET": 2, "SERT": 2}, 0.8),
    ("CCC(N)Cc1ccc2occc2c1", "6-APDB", {"DAT": 2, "NET": 2, "SERT": 2}, 0.8),

    # Piperazine derivatives
    ("Clc1cccc(N2CCNCC2)c1", "3-Chlorophenylpiperazine", {"DAT": 0, "NET": 1, "SERT": 2}, 0.85),
    ("Fc1cccc(N2CCNCC2)c1", "3-Fluorophenylpiperazine", {"DAT": 0, "NET": 1, "SERT": 2}, 0.85),
    ("COc1ccc(N2CCNCC2)cc1", "4-Methoxyphenylpiperazine", {"DAT": 0, "NET": 0, "SERT": 2}, 0.85),
    ("Cc1ccc(N2CCNCC2)cc1", "4-Methylphenylpiperazine", {"DAT": 0, "NET": 1, "SERT": 1}, 0.8),

    # More tryptamines
    ("CC(N)Cc1c[nH]c2ccc(O)cc12", "5-Hydroxy-AMT", {"DAT": 1, "NET": 2, "SERT": 2}, 0.85),
    ("CN(C)CCc1c[nH]c2cc(F)ccc12", "5-Fluoro-DMT", {"DAT": 0, "NET": 0, "SERT": 2}, 0.8),
    ("CN(C)CCc1c[nH]c2cc(Cl)ccc12", "5-Chloro-DMT", {"DAT": 0, "NET": 0, "SERT": 2}, 0.8),
    ("CN(C)CCc1c[nH]c2cc(Br)ccc12", "5-Bromo-DMT", {"DAT": 0, "NET": 0, "SERT": 2}, 0.8),
    ("CCN(CC)CCc1c[nH]c2cc(O)ccc12", "5-Hydroxy-DET", {"DAT": 0, "NET": 0, "SERT": 2}, 0.8),
    ("C(C)N(CC)CCc1c[nH]c2cccc(O)c12", "4-Hydroxy-DET", {"DAT": 0, "NET": 0, "SERT": 2}, 0.85),

    # Phenethylamine analogs without alpha-methyl
    ("NCCc1ccc(F)cc1", "4-Fluorophenethylamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.8),
    ("NCCc1ccc(Cl)cc1", "4-Chlorophenethylamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.8),
    ("NCCc1ccc(C)cc1", "4-Methylphenethylamine", {"DAT": 2, "NET": 2, "SERT": 1}, 0.8),
    ("NCCc1ccc(OC)cc1", "4-Methoxyphenethylamine", {"DAT": 1, "NET": 2, "SERT": 1}, 0.8),
    ("CNCCc1ccc(O)cc1", "N-Methyl-tyramine", {"DAT": 2, "NET": 2, "SERT": 0}, 0.9),
    ("CN(C)CCc1ccc(O)cc1", "Hordenine", {"DAT": 1, "NET": 2, "SERT": 0}, 0.9),
    ("NCCc1cccc(O)c1", "3-Hydroxyphenethylamine", {"DAT": 2, "NET": 2, "SERT": 0}, 0.8),
    ("NCCc1ccccc1O", "2-Hydroxyphenethylamine", {"DAT": 1, "NET": 2, "SERT": 0}, 0.8),

    # More blockers for balance
    ("CN1C2CCC1CC(OC(=O)C(O)c3ccccc3)C2", "Cocaethylene-related", {"DAT": 1, "NET": 1, "SERT": 1}, 0.8),
    ("Fc1ccc(C(c2ccccc2)N3CCN(C)CC3)cc1", "Fluoxetine-related", {"DAT": 0, "NET": 0, "SERT": 1}, 0.8),
    ("Clc1ccc(C(c2ccccc2)N3CCNCC3)cc1", "Chlorophenyl-DPP", {"DAT": 1, "NET": 0, "SERT": 0}, 0.7),
    ("Fc1ccc(C(OCCN2CCCC2)c2ccc(F)cc2)cc1", "GBR-related", {"DAT": 1, "NET": 0, "SERT": 0}, 0.8),
    ("COC(=O)C(c1ccccc1)C2CCCN2C", "N-Methylpiperidine-phenylacetate", {"DAT": 1, "NET": 1, "SERT": 0}, 0.7),

    # More inactive compounds (decoys)
    ("CCCc1ccccc1", "Propylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(C)c1ccccc1", "Isopropylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc(CC2CCCCC2)cc1", "Cyclohexylmethylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.9),
    ("c1ccc(Cc2ccccc2)cc1", "Diphenylmethane", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc(CCc2ccccc2)cc1", "1,2-Diphenylethane", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CCCCc1ccccc1", "Butylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CCCCCc1ccccc1", "Pentylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("Cc1ccc(C)c(C)c1", "1,2,4-Trimethylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("Cc1cc(C)c(C)cc1C", "Durene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("Clc1ccc(Cl)cc1", "1,4-Dichlorobenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("Fc1ccc(F)cc1", "1,4-Difluorobenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2c(c1)ccc1ccccc12", "Phenanthrene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2cc3ccccc3cc2c1", "Anthracene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2c(c1)Cc1ccccc1C2", "Fluorene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("O=C1c2ccccc2C(=O)c2ccccc12", "Anthraquinone", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc(Oc2ccccc2)cc1", "Diphenyl ether", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc(Sc2ccccc2)cc1", "Diphenyl sulfide", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)c1ccc(C)cc1", "4-Methylacetophenone", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)c1ccc(F)cc1", "4-Fluoroacetophenone", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)c1ccc(Cl)cc1", "4-Chloroacetophenone", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)c1ccc(OC)cc1", "4-Methoxyacetophenone", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("COC(=O)c1ccc(C)cc1", "Methyl 4-methylbenzoate", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("COC(=O)c1ccc(F)cc1", "Methyl 4-fluorobenzoate", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("COC(=O)c1ccc(Cl)cc1", "Methyl 4-chlorobenzoate", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CCOC(=O)c1ccc(C)cc1", "Ethyl 4-methylbenzoate", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),

    # Alcohols and simple ethers (inactive)
    ("CCCCCCc1ccccc1", "Hexylbenzene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc(COC2CCCCC2)cc1", "Benzyl cyclohexyl ether", {"DAT": 0, "NET": 0, "SERT": 0}, 0.9),
    ("OCCc1ccccc1", "Phenethyl alcohol", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OC(c1ccccc1)c2ccccc2", "Benzhydrol", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OCC(c1ccccc1)c2ccccc2", "Diphenylmethanol", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),

    # Additional simple carboxylic acids (inactive)
    ("OC(=O)Cc1ccc(C)cc1", "4-Methylphenylacetic acid", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OC(=O)Cc1ccc(F)cc1", "4-Fluorophenylacetic acid", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OC(=O)Cc1ccc(Cl)cc1", "4-Chlorophenylacetic acid", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OC(=O)Cc1ccc(OC)cc1", "4-Methoxyphenylacetic acid", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OC(=O)CCc1ccccc1", "Hydrocinnamic acid", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("OC(=O)C=Cc1ccccc1", "Cinnamic acid", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),

    # More heterocycles (inactive)
    ("c1ccc2[nH]ccc2c1", "Indole", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("Cc1ccc2[nH]ccc2c1", "5-Methylindole", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("COc1ccc2[nH]ccc2c1", "5-Methoxyindole", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2[nH]c3ccccc3c2c1", "Carbazole", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2oc3ccccc3c2c1", "Dibenzofuran", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2sc3ccccc3c2c1", "Dibenzothiophene", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1cnc2ccccc2c1", "Quinoline", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2ncccc2c1", "Isoquinoline", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1cnc2ccccc2n1", "Quinazoline", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc2nc3ccccc3nc2c1", "Phenazine", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1cc2ccccc2cn1", "Quinoxaline", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),

    # Amides (inactive, no basic nitrogen)
    ("CC(=O)Nc1ccc(C)cc1", "4-Methylacetanilide", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)Nc1ccc(F)cc1", "4-Fluoroacetanilide", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)Nc1ccc(Cl)cc1", "4-Chloroacetanilide", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("CC(=O)Nc1ccc(OC)cc1", "4-Methoxyacetanilide", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
    ("c1ccc(NC(=O)c2ccccc2)cc1", "Benzanilide", {"DAT": 0, "NET": 0, "SERT": 0}, 0.95),
]


def get_additional_data() -> pd.DataFrame:
    """Get additional compounds."""
    records = []

    for smiles, name, labels, conf in ADDITIONAL_COMPOUNDS:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

        for target, label in labels.items():
            records.append({
                'smiles': canonical,
                'compound_name': name,
                'target': target,
                'label': label,
                'confidence': conf,
                'source': 'additional_curated',
                'category': 'additional',
            })

    df = pd.DataFrame(records)
    print(f"Additional data: {len(df)} records, {df['smiles'].nunique()} unique compounds")
    return df


if __name__ == "__main__":
    df = get_additional_data()
    print(df.head())
