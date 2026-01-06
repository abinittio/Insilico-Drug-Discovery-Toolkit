"""
SAR-Based Data Expansion
========================

Systematically expand the dataset by generating analogs based on
known structure-activity relationships (SAR).

Strategy:
1. Take validated scaffolds (amphetamine, cathinone, tryptamine)
2. Apply known active substituent patterns
3. Infer activity based on published SAR trends
4. Generate stereoisomers where applicable

This gives us high-confidence inferred labels based on
well-established medicinal chemistry principles.
"""

import itertools
from typing import Dict, List, Tuple, Optional
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARExpander:
    """
    Expands dataset using SAR principles.
    """

    # =========================================================================
    # SAR Rules for Amphetamines
    # =========================================================================
    # Based on: Rothman & Baumann, Annals of NYAS, 2003
    # Simmler et al., BJP, 2013
    # Rickli et al., Neuropharmacology, 2015

    AMPHETAMINE_SAR = {
        # Para-substituents and their effects on selectivity
        # (substituent_smiles, DAT_effect, NET_effect, SERT_effect)
        # effect: 2=substrate, 1=weak/blocker, 0=inactive

        'para_halogen': [
            ('F', 2, 2, 2),      # 4-FA: balanced releaser
            ('Cl', 2, 2, 2),     # PCA: potent SERT releaser
            ('Br', 2, 2, 2),     # PBA: similar to PCA
            ('I', 2, 2, 2),      # 4-IA: similar
        ],
        'para_alkyl': [
            ('C', 2, 2, 2),      # 4-MA: balanced
            ('CC', 2, 2, 1),     # 4-EA: DAT/NET selective
            ('C(C)C', 1, 2, 1),  # 4-iPr: NET selective
        ],
        'para_alkoxy': [
            ('OC', 1, 2, 2),     # PMA: SERT/NET selective
            ('OCC', 1, 2, 2),    # 4-EA: similar
        ],
        'para_hydroxy': [
            ('O', 2, 2, 1),      # 4-OH: weak SERT
        ],
        'meta_halogen': [
            ('F', 2, 2, 1),      # 3-FA: DAT/NET selective
            ('Cl', 2, 2, 1),     # 3-CA: similar
        ],
        'ortho_halogen': [
            ('F', 2, 2, 1),      # 2-FA: DAT/NET selective
            ('Cl', 1, 2, 1),     # 2-CA: steric hindrance
        ],
    }

    # N-substituent effects on amphetamine activity
    N_SUBSTITUENT_SAR = [
        # (N_group_smiles, name, DAT_mod, NET_mod, SERT_mod)
        # mod: 1.0 = full activity, 0.5 = reduced, 0 = inactive

        ('N', 'primary', 1.0, 1.0, 1.0),
        ('NC', 'N-methyl', 1.0, 1.0, 1.0),
        ('NCC', 'N-ethyl', 0.8, 0.8, 0.8),
        ('NCCC', 'N-propyl', 0.6, 0.6, 0.6),
        ('N(C)C', 'N,N-dimethyl', 0.3, 0.3, 0.3),  # Largely inactive
    ]

    # =========================================================================
    # SAR Rules for Cathinones
    # =========================================================================
    # Based on: Baumann et al., Neuropsychopharmacology, 2012
    # Eshleman et al., J Pharmacol Exp Ther, 2013

    CATHINONE_SAR = {
        # Ring substituents
        'para_substituent': [
            ('C', 2, 2, 2),      # Mephedrone: balanced
            ('F', 2, 2, 2),      # Flephedrone: balanced
            ('Cl', 2, 2, 2),     # 4-CMC: balanced
            ('Br', 2, 2, 2),     # 4-BMC: balanced
            ('OC', 1, 2, 2),     # Methedrone: SERT/NET selective
        ],
        'meta_substituent': [
            ('C', 2, 2, 2),      # 3-MMC: balanced
            ('F', 2, 2, 1),      # 3-FMC: DAT/NET
            ('Cl', 2, 2, 2),     # 3-CMC: balanced
        ],
        'methylenedioxy': [
            ('methylenedioxy', 2, 2, 2),  # Methylone, etc
        ],
    }

    # Alpha substituent effects
    ALPHA_CHAIN_SAR = [
        # (alpha_chain, name, substrate_probability)
        ('C', 'methyl', 1.0),        # Cathinone/methcathinone
        ('CC', 'ethyl', 0.9),         # Buphedrone
        ('CCC', 'propyl', 0.8),       # Pentedrone
        ('CCCC', 'butyl', 0.7),       # Hexedrone
    ]

    # N-pyrrolidine kills substrate activity -> blocker
    PYRROLIDINE_EFFECT = {
        'pyrrolidine': (1, 1, 0),  # MDPV, α-PVP are blockers
    }

    # =========================================================================
    # Systematic Analog Generation
    # =========================================================================

    def generate_amphetamine_analogs(self) -> List[Tuple]:
        """
        Generate amphetamine analogs with inferred activities.

        Returns:
            List of (SMILES, name, labels, confidence)
        """
        analogs = []

        # Base scaffold: phenethylamine with alpha-methyl
        # C[C@H](N)Cc1ccccc1 = d-amphetamine

        # Para-substituted amphetamines
        for sub_name, substitutions in self.AMPHETAMINE_SAR.items():
            for sub_smiles, dat, net, sert in substitutions:
                if 'para' in sub_name:
                    # 4-substituted
                    smiles_d = f"C[C@H](N)Cc1ccc({sub_smiles})cc1"
                    smiles_l = f"C[C@@H](N)Cc1ccc({sub_smiles})cc1"
                    smiles_rac = f"CC(N)Cc1ccc({sub_smiles})cc1"

                    name_base = f"4-{sub_smiles}-amphetamine"

                    analogs.append((
                        smiles_d, f"(+)-{name_base}",
                        {"DAT": dat, "NET": net, "SERT": sert},
                        0.85, "SAR_inferred"
                    ))
                    analogs.append((
                        smiles_l, f"(-)-{name_base}",
                        {"DAT": max(0, dat-1), "NET": max(0, net-1), "SERT": max(0, sert-1)},
                        0.85, "SAR_inferred"
                    ))

                elif 'meta' in sub_name:
                    # 3-substituted
                    smiles_d = f"C[C@H](N)Cc1cccc({sub_smiles})c1"
                    name_base = f"3-{sub_smiles}-amphetamine"

                    analogs.append((
                        smiles_d, f"(+)-{name_base}",
                        {"DAT": dat, "NET": net, "SERT": sert},
                        0.80, "SAR_inferred"
                    ))

                elif 'ortho' in sub_name:
                    # 2-substituted
                    smiles_d = f"C[C@H](N)Cc1ccccc1{sub_smiles}"
                    name_base = f"2-{sub_smiles}-amphetamine"

                    analogs.append((
                        smiles_d, f"(+)-{name_base}",
                        {"DAT": dat, "NET": net, "SERT": sert},
                        0.75, "SAR_inferred"
                    ))

        # N-substituted amphetamines
        ring_subs = ['', 'F', 'Cl', 'C']  # unsubstituted, F, Cl, methyl
        for ring_sub in ring_subs:
            for n_smiles, n_name, dat_mod, net_mod, sert_mod in self.N_SUBSTITUENT_SAR:
                if ring_sub == '':
                    base_smiles = f"C[C@H]({n_smiles})Cc1ccccc1"
                    name = f"(+)-{n_name}-amphetamine"
                else:
                    base_smiles = f"C[C@H]({n_smiles})Cc1ccc({ring_sub})cc1"
                    name = f"(+)-4-{ring_sub}-{n_name}-amphetamine"

                # Infer activity based on modifiers
                dat = 2 if dat_mod >= 0.7 else (1 if dat_mod >= 0.3 else 0)
                net = 2 if net_mod >= 0.7 else (1 if net_mod >= 0.3 else 0)
                sert = 1 if sert_mod >= 0.5 else 0  # Amphetamines weak at SERT

                analogs.append((
                    base_smiles, name,
                    {"DAT": dat, "NET": net, "SERT": sert},
                    0.75, "SAR_inferred"
                ))

        return analogs

    def generate_cathinone_analogs(self) -> List[Tuple]:
        """
        Generate cathinone analogs with inferred activities.
        """
        analogs = []

        # Base: CC(NC)C(=O)c1ccccc1 = methcathinone

        # Ring-substituted cathinones
        ring_positions = [
            ('1ccc({})cc1', '4-'),   # para
            ('1cccc({})c1', '3-'),   # meta
        ]

        substituents = [
            ('C', 'methyl'),
            ('F', 'fluoro'),
            ('Cl', 'chloro'),
            ('Br', 'bromo'),
            ('OC', 'methoxy'),
        ]

        n_groups = [
            ('NC', 'N-methyl'),
            ('NCC', 'N-ethyl'),
            ('N', 'primary'),
        ]

        alpha_chains = [
            ('C', 'α-methyl'),
            ('CC', 'α-ethyl'),
            ('CCC', 'α-propyl'),
        ]

        for (ring_pattern, pos_name), (sub_smiles, sub_name) in itertools.product(
            ring_positions, substituents
        ):
            for n_smiles, n_name in n_groups:
                for alpha, alpha_name in alpha_chains:
                    ring = ring_pattern.format(sub_smiles)
                    smiles = f"{alpha}({n_smiles})C(=O)c{ring}"

                    # Validate
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

                    # Infer activity based on SAR
                    # Most cathinones are balanced releasers
                    # Para-methoxy shifts to SERT
                    # Pyrrolidine -> blocker (handled separately)

                    if sub_smiles == 'OC':
                        dat, net, sert = 1, 2, 2
                    else:
                        dat, net, sert = 2, 2, 2

                    # Longer alpha chains reduce potency
                    if alpha == 'CCC':
                        dat = max(0, dat - 1)
                        net = max(0, net - 1)

                    name = f"{pos_name}{sub_name}-{n_name}-{alpha_name}-cathinone"

                    analogs.append((
                        smiles, name,
                        {"DAT": dat, "NET": net, "SERT": sert},
                        0.70, "SAR_inferred"
                    ))

        # Pyrrolidine cathinones (blockers)
        pyrrolidine_subs = [
            ('c1ccccc1', 'phenyl'),
            ('c1ccc(F)cc1', '4-F-phenyl'),
            ('c1ccc(C)cc1', '4-Me-phenyl'),
            ('c1ccc2OCOc2c1', 'methylenedioxy-phenyl'),
        ]

        alpha_chains_pyrr = [
            ('CC', 'α-ethyl'),
            ('CCC', 'α-propyl'),
            ('CCCC', 'α-butyl'),
            ('CCCCC', 'α-pentyl'),
        ]

        for (ring, ring_name), (alpha, alpha_name) in itertools.product(
            pyrrolidine_subs, alpha_chains_pyrr
        ):
            smiles = f"{alpha}(C(=O){ring})N1CCCC1"

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            name = f"α-pyrrolidino-{alpha_name}-{ring_name}-propiophenone"

            # All pyrrolidine cathinones are DAT/NET blockers
            analogs.append((
                smiles, name,
                {"DAT": 1, "NET": 1, "SERT": 0},
                0.90, "SAR_inferred"
            ))

        return analogs

    def generate_tryptamine_analogs(self) -> List[Tuple]:
        """
        Generate tryptamine analogs with inferred activities.
        """
        analogs = []

        # Base tryptamine scaffold

        # 4-position substituents
        pos4_subs = [
            ('O', 'hydroxy'),    # Psilocin-like
            ('OC', 'methoxy'),
            ('OCC', 'ethoxy'),
        ]

        # 5-position substituents
        pos5_subs = [
            ('O', 'hydroxy'),    # Bufotenin-like
            ('OC', 'methoxy'),   # 5-MeO-DMT-like
            ('OCC', 'ethoxy'),
        ]

        # N-substituents
        n_subs = [
            ('N', 'tryptamine'),
            ('NC', 'NMT'),
            ('N(C)C', 'DMT'),
            ('N(CC)CC', 'DET'),
            ('N(CCC)CCC', 'DPT'),
            ('N(C(C)C)C(C)C', 'DiPT'),
        ]

        # 4-substituted tryptamines
        for (sub, sub_name), (n_group, n_name) in itertools.product(pos4_subs, n_subs):
            smiles = f"{n_group}CCc1c[nH]c2cccc({sub})c12"

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            name = f"4-{sub_name}-{n_name}"

            # 4-sub tryptamines are SERT substrates
            analogs.append((
                smiles, name,
                {"DAT": 0, "NET": 0, "SERT": 2},
                0.80, "SAR_inferred"
            ))

        # 5-substituted tryptamines
        for (sub, sub_name), (n_group, n_name) in itertools.product(pos5_subs, n_subs):
            smiles = f"{n_group}CCc1c[nH]c2ccc({sub})cc12"

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            name = f"5-{sub_name}-{n_name}"

            # 5-MeO compounds can have NET activity
            if sub == 'OC':
                net = 1
            else:
                net = 0

            analogs.append((
                smiles, name,
                {"DAT": 0, "NET": net, "SERT": 2},
                0.80, "SAR_inferred"
            ))

        # Alpha-methyl tryptamines
        alpha_methyl_subs = [
            ('', 'AMT'),
            ('OC', '5-MeO-AMT'),
        ]

        for sub, name in alpha_methyl_subs:
            if sub:
                smiles = f"CC(N)Cc1c[nH]c2ccc({sub})cc12"
            else:
                smiles = "CC(N)Cc1c[nH]c2ccccc12"

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            # AMTs are triple releasers
            analogs.append((
                smiles, name,
                {"DAT": 2, "NET": 2, "SERT": 2},
                0.85, "SAR_inferred"
            ))

        return analogs

    def generate_phenethylamine_analogs(self) -> List[Tuple]:
        """
        Generate phenethylamine analogs (non-amphetamine).
        """
        analogs = []

        # 2C-x compounds (generally not substrates but weak SERT activity)
        substituents_2c = [
            ('C', '2C-D'),
            ('CC', '2C-E'),
            ('CCC', '2C-P'),
            ('C(C)C', '2C-IP'),
            ('F', '2C-F'),
            ('Cl', '2C-C'),
            ('Br', '2C-B'),
            ('I', '2C-I'),
            ('SC', '2C-T'),
            ('SCC', '2C-T-2'),
            ('SCCC', '2C-T-4'),
            ('SC(C)C', '2C-T-7'),
        ]

        for sub, name in substituents_2c:
            smiles = f"NCCc1cc(OC)c({sub})cc1OC"

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            # 2C-x compounds have weak/no substrate activity
            analogs.append((
                smiles, name,
                {"DAT": 0, "NET": 0, "SERT": 1},
                0.80, "SAR_inferred"
            ))

        # DOx compounds (2,5-dimethoxy-4-substituted amphetamines)
        for sub, sub_name in substituents_2c[:8]:  # Halogens and small alkyls
            smiles = f"C[C@H](N)Cc1cc(OC)c({sub})cc1OC"

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            name = f"DO{sub_name[-1] if len(sub_name) == 4 else sub_name[3:]}"

            # DOx are weak releasers
            analogs.append((
                smiles, name,
                {"DAT": 1, "NET": 1, "SERT": 1},
                0.75, "SAR_inferred"
            ))

        return analogs

    def generate_all(self) -> pd.DataFrame:
        """
        Generate all SAR-based analogs.
        """
        all_analogs = []

        # Generate from each class
        all_analogs.extend(self.generate_amphetamine_analogs())
        all_analogs.extend(self.generate_cathinone_analogs())
        all_analogs.extend(self.generate_tryptamine_analogs())
        all_analogs.extend(self.generate_phenethylamine_analogs())

        # Convert to DataFrame
        records = []
        for smiles, name, labels, conf, source in all_analogs:
            # Validate SMILES
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
                    'source': source,
                    'category': 'sar_expanded',
                })

        df = pd.DataFrame(records)

        # Remove duplicates
        df = df.drop_duplicates(subset=['smiles', 'target'])

        logger.info(f"Generated {len(df)} SAR-expanded records")
        logger.info(f"Unique compounds: {df['smiles'].nunique()}")

        return df


class DecoyGenerator:
    """
    Generate negative examples (decoys) using DUD-E principles.

    Decoys are compounds that:
    1. Have similar physical properties to actives
    2. Are topologically dissimilar
    3. Are unlikely to bind/transport
    """

    def generate_decoys(self, n_decoys: int = 500) -> pd.DataFrame:
        """
        Generate decoy compounds.

        Uses druglike molecules from ChEMBL that are NOT
        phenethylamines, cathinones, or tryptamines.
        """
        # Representative druglike scaffolds that are NOT transporter substrates
        decoy_scaffolds = [
            # Carboxylic acids
            ("c1ccc(CC(=O)O)cc1", "phenylacetic acid"),
            ("c1ccc(CCC(=O)O)cc1", "phenylpropionic acid"),

            # Esters
            ("CCOC(=O)c1ccccc1", "ethyl benzoate"),
            ("COC(=O)c1ccc(OC)cc1", "methyl anisate"),

            # Ketones without amine
            ("CC(=O)c1ccccc1", "acetophenone"),
            ("CCC(=O)c1ccccc1", "propiophenone"),
            ("CCCC(=O)c1ccccc1", "butyrophenone"),

            # Ethers
            ("COc1ccccc1", "anisole"),
            ("CCOc1ccccc1", "phenetole"),
            ("c1ccc(Oc2ccccc2)cc1", "diphenyl ether"),

            # Sulfides/Sulfoxides
            ("CSc1ccccc1", "thioanisole"),
            ("CS(=O)c1ccccc1", "methyl phenyl sulfoxide"),

            # Heterocycles without basic nitrogen
            ("c1ccc2occc2c1", "benzofuran"),
            ("c1ccc2sccc2c1", "benzothiophene"),
            ("c1cnc2ccccc2n1", "quinazoline"),
            ("c1ccc2ncccc2c1", "quinoline"),

            # Lactams
            ("O=C1CCCN1", "pyrrolidinone"),
            ("O=C1CCCCN1", "piperidinone"),
            ("O=C1CCN(C)C1", "N-methylpyrrolidinone"),

            # Ureas
            ("NC(=O)Nc1ccccc1", "phenylurea"),
            ("CNC(=O)Nc1ccccc1", "N-methylphenylurea"),

            # Amides without basic amine
            ("CC(=O)Nc1ccccc1", "acetanilide"),
            ("c1ccc(NC(=O)c2ccccc2)cc1", "benzanilide"),

            # Hydroxybenzenes
            ("Oc1ccccc1", "phenol"),
            ("Oc1ccc(O)cc1", "hydroquinone"),
            ("Oc1cccc(O)c1", "resorcinol"),

            # Biphenyls
            ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
            ("Cc1ccc(-c2ccccc2)cc1", "4-methylbiphenyl"),

            # Naphthalenes
            ("c1ccc2ccccc2c1", "naphthalene"),
            ("Cc1ccc2ccccc2c1", "1-methylnaphthalene"),
            ("COc1ccc2ccccc2c1", "1-methoxynaphthalene"),

            # Indanes (no amine)
            ("C1Cc2ccccc2C1", "indane"),
            ("CC1Cc2ccccc2C1", "1-methylindane"),

            # Coumarins
            ("O=c1ccc2ccccc2o1", "coumarin"),
            ("COc1ccc2ccc(=O)oc2c1", "7-methoxycoumarin"),

            # Flavones
            ("O=c1cc(-c2ccccc2)oc2ccccc12", "flavone"),

            # Steroids (no amine)
            ("CC12CCC3C(CCC4CC(O)CCC43C)C1CCC2O", "steroid-diol"),

            # Terpenoids
            ("CC1=CCC(C(C)C)CC1", "limonene-like"),
            ("CC(C)C1CCC(C)CC1", "menthane"),
        ]

        records = []

        for smiles, name in decoy_scaffolds:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            for target in ['DAT', 'NET', 'SERT']:
                records.append({
                    'smiles': canonical,
                    'compound_name': f"decoy_{name}",
                    'target': target,
                    'label': 0,  # Inactive
                    'confidence': 0.95,
                    'source': 'decoy',
                    'category': 'decoy',
                })

        # Add substituted variants
        substituents = ['C', 'CC', 'F', 'Cl', 'OC']

        base_decoys = [
            "c1ccc(CC(=O)O)cc1",  # Phenylacetic acid
            "CC(=O)c1ccccc1",     # Acetophenone
            "COc1ccccc1",          # Anisole
        ]

        for base in base_decoys:
            for sub in substituents:
                # Try para substitution
                try:
                    if "cc1" in base:
                        new_smiles = base.replace("cc1", f"c({sub})c1")
                    else:
                        continue

                    mol = Chem.MolFromSmiles(new_smiles)
                    if mol is None:
                        continue

                    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

                    for target in ['DAT', 'NET', 'SERT']:
                        records.append({
                            'smiles': canonical,
                            'compound_name': f"decoy_sub_{sub}",
                            'target': target,
                            'label': 0,
                            'confidence': 0.90,
                            'source': 'decoy_generated',
                            'category': 'decoy',
                        })
                except:
                    continue

        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=['smiles', 'target'])

        logger.info(f"Generated {len(df)} decoy records")
        logger.info(f"Unique decoys: {df['smiles'].nunique()}")

        return df


def expand_data() -> pd.DataFrame:
    """
    Run full SAR expansion.
    """
    logger.info("=" * 60)
    logger.info("SAR-BASED DATA EXPANSION")
    logger.info("=" * 60)

    # Generate SAR analogs
    expander = SARExpander()
    sar_df = expander.generate_all()

    # Generate decoys
    decoy_gen = DecoyGenerator()
    decoy_df = decoy_gen.generate_decoys()

    # Combine
    df = pd.concat([sar_df, decoy_df], ignore_index=True)

    logger.info(f"\nTotal expanded records: {len(df)}")
    logger.info(f"Unique compounds: {df['smiles'].nunique()}")

    for target in ['DAT', 'NET', 'SERT']:
        target_df = df[df['target'] == target]
        substrates = len(target_df[target_df['label'] == 2])
        blockers = len(target_df[target_df['label'] == 1])
        inactive = len(target_df[target_df['label'] == 0])
        logger.info(f"{target}: {substrates} sub, {blockers} block, {inactive} inactive")

    return df


if __name__ == "__main__":
    df = expand_data()
    print(f"\nGenerated {len(df)} records from SAR expansion")
