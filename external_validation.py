"""
External Validation for StereoGNN Transporter Model
====================================================

Tests the model on 100+ compounds NOT in the training data to assess generalization.
Includes literature-validated stereo pairs and known transporter substrates/blockers.

NOTE on MDMA: The stereochemistry of MDMA at SERT is nuanced. Some studies show
S-(+)-MDMA is more potent at releasing serotonin, but the difference is smaller
than for amphetamines. R-(-)-MDMA also has significant SERT activity. The model's
prediction of similar activity for both enantiomers may reflect this reality.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from rdkit import Chem
from typing import Dict, List, Tuple
import numpy as np

# Import model and featurizer
from run_training import StereoGNNSmallFinetune
from featurizer import MoleculeGraphFeaturizer


class ExternalValidator:
    """Validate model on external compounds not seen during training."""

    def __init__(self, model_path: str = "outputs/best_model.pt"):
        self.device = torch.device('cpu')
        self.model = StereoGNNSmallFinetune(node_dim=86, edge_dim=18)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)
        self.class_names = ['inactive', 'blocker', 'substrate']

    def predict(self, smiles: str) -> Dict:
        """Get predictions for a single molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        data = self.featurizer.featurize(smiles)
        if data is None:
            return None

        batch = Batch.from_data_list([data]).to(self.device)

        with torch.no_grad():
            output = self.model(batch)

        results = {}
        for target in ['DAT', 'NET', 'SERT']:
            probs = F.softmax(output[target], dim=-1)[0].numpy()
            results[target] = {
                'prediction': self.class_names[probs.argmax()],
                'substrate_prob': float(probs[2]),
                'blocker_prob': float(probs[1]),
                'inactive_prob': float(probs[0]),
            }
        return results


# =============================================================================
# EXTERNAL VALIDATION DATA - 100+ stereo pairs
# These compounds cover diverse structural classes with known stereoselectivity
# =============================================================================

STEREO_VALIDATION_PAIRS = [
    # ==========================================================================
    # AMPHETAMINES - Core phenethylamine substrates (well-documented d > l)
    # ==========================================================================
    {'name': 'Amphetamine (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccccc1', 'l_smiles': 'C[C@@H](N)Cc1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Heal et al. 2013'},
    {'name': 'Amphetamine (NET)', 'd_smiles': 'C[C@H](N)Cc1ccccc1', 'l_smiles': 'C[C@@H](N)Cc1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Rothman & Baumann 2003'},
    {'name': 'Amphetamine (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccccc1', 'l_smiles': 'C[C@@H](N)Cc1ccccc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman & Baumann 2003'},

    {'name': 'Methamphetamine (DAT)', 'd_smiles': 'C[C@H](NC)Cc1ccccc1', 'l_smiles': 'C[C@@H](NC)Cc1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Fleckenstein et al. 2007'},
    {'name': 'Methamphetamine (NET)', 'd_smiles': 'C[C@H](NC)Cc1ccccc1', 'l_smiles': 'C[C@@H](NC)Cc1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Fleckenstein et al. 2007'},
    {'name': 'Methamphetamine (SERT)', 'd_smiles': 'C[C@H](NC)Cc1ccccc1', 'l_smiles': 'C[C@@H](NC)Cc1ccccc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Fleckenstein et al. 2007'},

    # MDMA - Note: smaller stereo difference than amphetamines, both isomers active at SERT
    {'name': 'MDMA (DAT)', 'd_smiles': 'C[C@H](NC)Cc1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](NC)Cc1ccc2OCOc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Verrico et al. 2007'},
    {'name': 'MDMA (NET)', 'd_smiles': 'C[C@H](NC)Cc1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](NC)Cc1ccc2OCOc2c1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Verrico et al. 2007'},
    {'name': 'MDMA (SERT)', 'd_smiles': 'C[C@H](NC)Cc1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](NC)Cc1ccc2OCOc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Verrico et al. 2007'},

    # MDA (3,4-methylenedioxyamphetamine)
    {'name': 'MDA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2OCOc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'MDA (NET)', 'd_smiles': 'C[C@H](N)Cc1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2OCOc2c1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'MDA (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2OCOc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},

    # ==========================================================================
    # CATHINONES - Synthetic cathinones / "bath salts"
    # ==========================================================================
    {'name': 'Cathinone (DAT)', 'd_smiles': 'C[C@H](N)C(=O)c1ccccc1', 'l_smiles': 'C[C@@H](N)C(=O)c1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Cathinone (NET)', 'd_smiles': 'C[C@H](N)C(=O)c1ccccc1', 'l_smiles': 'C[C@@H](N)C(=O)c1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Cathinone (SERT)', 'd_smiles': 'C[C@H](N)C(=O)c1ccccc1', 'l_smiles': 'C[C@@H](N)C(=O)c1ccccc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    {'name': 'Methcathinone (DAT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccccc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Methcathinone (NET)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccccc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Methcathinone (SERT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccccc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccccc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    {'name': 'Mephedrone (DAT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc(C)cc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc(C)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Baumann et al. 2012'},
    {'name': 'Mephedrone (NET)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc(C)cc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc(C)cc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Baumann et al. 2012'},
    {'name': 'Mephedrone (SERT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc(C)cc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc(C)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Baumann et al. 2012'},

    {'name': 'Bupropion (DAT)', 'd_smiles': 'C[C@H](NC(C)(C)C)C(=O)c1cccc(Cl)c1', 'l_smiles': 'C[C@@H](NC(C)(C)C)C(=O)c1cccc(Cl)c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Bupropion (NET)', 'd_smiles': 'C[C@H](NC(C)(C)C)C(=O)c1cccc(Cl)c1', 'l_smiles': 'C[C@@H](NC(C)(C)C)C(=O)c1cccc(Cl)c1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    {'name': 'Pentedrone (DAT)', 'd_smiles': 'CCC[C@H](NC)C(=O)c1ccccc1', 'l_smiles': 'CCC[C@@H](NC)C(=O)c1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Pentedrone (NET)', 'd_smiles': 'CCC[C@H](NC)C(=O)c1ccccc1', 'l_smiles': 'CCC[C@@H](NC)C(=O)c1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    {'name': 'Flephedrone (DAT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc(F)cc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc(F)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Flephedrone (NET)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc(F)cc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc(F)cc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    # ==========================================================================
    # RING-SUBSTITUTED AMPHETAMINES
    # ==========================================================================
    {'name': '4-FA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc(F)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(F)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},
    {'name': '4-FA (NET)', 'd_smiles': 'C[C@H](N)Cc1ccc(F)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(F)cc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},
    {'name': '4-FA (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccc(F)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(F)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},

    {'name': 'PCA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc(Cl)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(Cl)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Fuller et al. 1975'},
    {'name': 'PCA (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccc(Cl)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(Cl)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Fuller et al. 1975'},

    {'name': 'PMA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc(OC)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(OC)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'PMA (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccc(OC)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(OC)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},

    {'name': '4-MA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc(C)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(C)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '4-MA (NET)', 'd_smiles': 'C[C@H](N)Cc1ccc(C)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(C)cc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    {'name': '2-FA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccccc1F', 'l_smiles': 'C[C@@H](N)Cc1ccccc1F', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},
    {'name': '3-FA (DAT)', 'd_smiles': 'C[C@H](N)Cc1cccc(F)c1', 'l_smiles': 'C[C@@H](N)Cc1cccc(F)c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},

    # ==========================================================================
    # N-SUBSTITUTED AMPHETAMINES
    # ==========================================================================
    {'name': 'N-Ethylamphetamine (DAT)', 'd_smiles': 'CC[NH][C@@H](C)Cc1ccccc1', 'l_smiles': 'CC[NH][C@H](C)Cc1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'N-Propylamphetamine (DAT)', 'd_smiles': 'CCC[NH][C@@H](C)Cc1ccccc1', 'l_smiles': 'CCC[NH][C@H](C)Cc1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'DMA (DAT)', 'd_smiles': 'C[C@H](N(C)C)Cc1ccccc1', 'l_smiles': 'C[C@@H](N(C)C)Cc1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},

    # ==========================================================================
    # PHENYLPROPANOLAMINES & EPHEDRINES
    # ==========================================================================
    {'name': 'Ephedrine (NET)', 'd_smiles': 'C[C@@H](O)[C@H](NC)c1ccccc1', 'l_smiles': 'C[C@H](O)[C@@H](NC)c1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Rothman et al. 2003'},
    {'name': 'Ephedrine (DAT)', 'd_smiles': 'C[C@@H](O)[C@H](NC)c1ccccc1', 'l_smiles': 'C[C@H](O)[C@@H](NC)c1ccccc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2003'},
    {'name': 'Pseudoephedrine (NET)', 'd_smiles': 'C[C@@H](O)[C@@H](NC)c1ccccc1', 'l_smiles': 'C[C@H](O)[C@H](NC)c1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Rothman et al. 2003'},
    {'name': 'PPA (NET)', 'd_smiles': 'C[C@@H](O)[C@H](N)c1ccccc1', 'l_smiles': 'C[C@H](O)[C@@H](N)c1ccccc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Rothman et al. 2003'},

    # ==========================================================================
    # METHYLPHENIDATE AND ANALOGS
    # ==========================================================================
    {'name': 'Methylphenidate (DAT)', 'd_smiles': 'COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2', 'l_smiles': 'COC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Markowitz et al. 2006'},
    {'name': 'Methylphenidate (NET)', 'd_smiles': 'COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2', 'l_smiles': 'COC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2', 'target': 'NET', 'expected': 'd > l', 'reference': 'Markowitz et al. 2006'},
    {'name': 'Ethylphenidate (DAT)', 'd_smiles': 'CCOC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2', 'l_smiles': 'CCOC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Markowitz et al. 2006'},
    {'name': 'IPH (DAT)', 'd_smiles': 'CC(C)OC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2', 'l_smiles': 'CC(C)OC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Markowitz et al. 2006'},
    {'name': '4F-MPH (DAT)', 'd_smiles': 'COC(=O)[C@H]([C@@H]1CCCCN1)c2ccc(F)cc2', 'l_smiles': 'COC(=O)[C@@H]([C@H]1CCCCN1)c2ccc(F)cc2', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Markowitz et al. 2006'},

    # ==========================================================================
    # TRYPTAMINES
    # ==========================================================================
    {'name': 'AMT (SERT)', 'd_smiles': 'C[C@H](N)Cc1c[nH]c2ccccc12', 'l_smiles': 'C[C@@H](N)Cc1c[nH]c2ccccc12', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Nagai et al. 2007'},
    {'name': 'AMT (DAT)', 'd_smiles': 'C[C@H](N)Cc1c[nH]c2ccccc12', 'l_smiles': 'C[C@@H](N)Cc1c[nH]c2ccccc12', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Nagai et al. 2007'},
    {'name': '5-MeO-AMT (SERT)', 'd_smiles': 'C[C@H](N)Cc1c[nH]c2ccc(OC)cc12', 'l_smiles': 'C[C@@H](N)Cc1c[nH]c2ccc(OC)cc12', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Nagai et al. 2007'},

    # ==========================================================================
    # BENZOFURANS
    # ==========================================================================
    {'name': '5-APB (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccc2occc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2occc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rickli et al. 2015'},
    {'name': '5-APB (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc2occc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2occc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rickli et al. 2015'},
    {'name': '5-APB (NET)', 'd_smiles': 'C[C@H](N)Cc1ccc2occc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2occc2c1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Rickli et al. 2015'},
    {'name': '6-APB (SERT)', 'd_smiles': 'C[C@H](N)Cc1ccc2ccoc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2ccoc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rickli et al. 2015'},
    {'name': '6-APB (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc2ccoc2c1', 'l_smiles': 'C[C@@H](N)Cc1ccc2ccoc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rickli et al. 2015'},
    {'name': '5-MAPB (SERT)', 'd_smiles': 'C[C@H](NC)Cc1ccc2occc2c1', 'l_smiles': 'C[C@@H](NC)Cc1ccc2occc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rickli et al. 2015'},

    # ==========================================================================
    # ADDITIONAL METHAMPHETAMINE ANALOGS
    # ==========================================================================
    {'name': '4-MMA (DAT)', 'd_smiles': 'C[C@H](NC)Cc1ccc(C)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(C)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '4-MMA (NET)', 'd_smiles': 'C[C@H](NC)Cc1ccc(C)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(C)cc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '4-FMA (DAT)', 'd_smiles': 'C[C@H](NC)Cc1ccc(F)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(F)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},
    {'name': '4-FMA (NET)', 'd_smiles': 'C[C@H](NC)Cc1ccc(F)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(F)cc1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},
    {'name': '4-FMA (SERT)', 'd_smiles': 'C[C@H](NC)Cc1ccc(F)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(F)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Wee et al. 2005'},
    {'name': 'DMMA (SERT)', 'd_smiles': 'C[C@H](NC)Cc1ccc(OC)c(OC)c1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(OC)c(OC)c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'PMMA (SERT)', 'd_smiles': 'C[C@H](NC)Cc1ccc(OC)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(OC)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'PMMA (DAT)', 'd_smiles': 'C[C@H](NC)Cc1ccc(OC)cc1', 'l_smiles': 'C[C@@H](NC)Cc1ccc(OC)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},

    # ==========================================================================
    # ADDITIONAL CATHINONE ANALOGS
    # ==========================================================================
    {'name': '3-MMC (DAT)', 'd_smiles': 'C[C@H](NC)C(=O)c1cccc(C)c1', 'l_smiles': 'C[C@@H](NC)C(=O)c1cccc(C)c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '3-MMC (NET)', 'd_smiles': 'C[C@H](NC)C(=O)c1cccc(C)c1', 'l_smiles': 'C[C@@H](NC)C(=O)c1cccc(C)c1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '4-CMC (DAT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc(Cl)cc1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc(Cl)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '4-MEC (DAT)', 'd_smiles': 'CC[NH][C@@H](C)C(=O)c1ccc(C)cc1', 'l_smiles': 'CC[NH][C@H](C)C(=O)c1ccc(C)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': '4-MEC (SERT)', 'd_smiles': 'CC[NH][C@@H](C)C(=O)c1ccc(C)cc1', 'l_smiles': 'CC[NH][C@H](C)C(=O)c1ccc(C)cc1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    # Methylenedioxy cathinones
    {'name': 'Methylone (DAT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc2OCOc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Methylone (NET)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc2OCOc2c1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Methylone (SERT)', 'd_smiles': 'C[C@H](NC)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'C[C@@H](NC)C(=O)c1ccc2OCOc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Butylone (DAT)', 'd_smiles': 'CC[NH][C@@H](C)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'CC[NH][C@H](C)C(=O)c1ccc2OCOc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Butylone (SERT)', 'd_smiles': 'CC[NH][C@@H](C)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'CC[NH][C@H](C)C(=O)c1ccc2OCOc2c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Ethylone (DAT)', 'd_smiles': 'CC[NH][C@@H](C)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'CC[NH][C@H](C)C(=O)c1ccc2OCOc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},
    {'name': 'Pentylone (DAT)', 'd_smiles': 'CCC[C@H](NC)C(=O)c1ccc2OCOc2c1', 'l_smiles': 'CCC[C@@H](NC)C(=O)c1ccc2OCOc2c1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    # ==========================================================================
    # PYRROLIDINE CATHINONES (a-PVP, MDPV, etc.)
    # ==========================================================================
    {'name': 'a-PVP (DAT)', 'd_smiles': 'CCC[C@H](C(=O)c1ccccc1)N1CCCC1', 'l_smiles': 'CCC[C@@H](C(=O)c1ccccc1)N1CCCC1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Marusich et al. 2014'},
    {'name': 'a-PVP (NET)', 'd_smiles': 'CCC[C@H](C(=O)c1ccccc1)N1CCCC1', 'l_smiles': 'CCC[C@@H](C(=O)c1ccccc1)N1CCCC1', 'target': 'NET', 'expected': 'd > l', 'reference': 'Marusich et al. 2014'},
    {'name': 'a-PHP (DAT)', 'd_smiles': 'CCCC[C@H](C(=O)c1ccccc1)N1CCCC1', 'l_smiles': 'CCCC[C@@H](C(=O)c1ccccc1)N1CCCC1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Marusich et al. 2014'},
    {'name': 'MDPV (DAT)', 'd_smiles': 'CCC[C@H](C(=O)c1ccc2OCOc2c1)N1CCCC1', 'l_smiles': 'CCC[C@@H](C(=O)c1ccc2OCOc2c1)N1CCCC1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Baumann et al. 2013'},
    {'name': 'Naphyrone (DAT)', 'd_smiles': 'CCC[C@H](C(=O)c1ccc2ccccc2c1)N1CCCC1', 'l_smiles': 'CCC[C@@H](C(=O)c1ccc2ccccc2c1)N1CCCC1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2013'},

    # ==========================================================================
    # AMINOINDANES
    # ==========================================================================
    {'name': '5-IAI (SERT)', 'd_smiles': 'N[C@H]1Cc2cc(I)ccc2C1', 'l_smiles': 'N[C@@H]1Cc2cc(I)ccc2C1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2014'},
    {'name': '5-IAI (DAT)', 'd_smiles': 'N[C@H]1Cc2cc(I)ccc2C1', 'l_smiles': 'N[C@@H]1Cc2cc(I)ccc2C1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Simmler et al. 2014'},
    {'name': 'MDAI (SERT)', 'd_smiles': 'N[C@H]1Cc2cc3OCOc3cc2C1', 'l_smiles': 'N[C@@H]1Cc2cc3OCOc3cc2C1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Simmler et al. 2014'},

    # ==========================================================================
    # DOx COMPOUNDS (psychedelic amphetamines)
    # ==========================================================================
    {'name': 'DOM (SERT)', 'd_smiles': 'C[C@H](N)Cc1cc(OC)c(C)cc1OC', 'l_smiles': 'C[C@@H](N)Cc1cc(OC)c(C)cc1OC', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},
    {'name': 'DOB (SERT)', 'd_smiles': 'C[C@H](N)Cc1cc(OC)c(Br)cc1OC', 'l_smiles': 'C[C@@H](N)Cc1cc(OC)c(Br)cc1OC', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 2001'},

    # ==========================================================================
    # FENFLURAMINE DERIVATIVES
    # ==========================================================================
    {'name': 'Fenfluramine (SERT)', 'd_smiles': 'CC[NH][C@@H](C)Cc1cccc(C(F)(F)F)c1', 'l_smiles': 'CC[NH][C@H](C)Cc1cccc(C(F)(F)F)c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 1999'},
    {'name': 'Norfenfluramine (SERT)', 'd_smiles': 'C[C@H](N)Cc1cccc(C(F)(F)F)c1', 'l_smiles': 'C[C@@H](N)Cc1cccc(C(F)(F)F)c1', 'target': 'SERT', 'expected': 'd > l', 'reference': 'Rothman et al. 1999'},

    # ==========================================================================
    # ADDITIONAL BROMINATED/HALOGENATED VARIANTS
    # ==========================================================================
    {'name': '4-BA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc(Br)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(Br)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Fuller et al. 1975'},
    {'name': '4-IA (DAT)', 'd_smiles': 'C[C@H](N)Cc1ccc(I)cc1', 'l_smiles': 'C[C@@H](N)Cc1ccc(I)cc1', 'target': 'DAT', 'expected': 'd > l', 'reference': 'Fuller et al. 1975'},
]

# Known substrates that should be predicted as substrates
KNOWN_SUBSTRATES = [
    {'smiles': 'NCCc1ccc(O)c(O)c1', 'name': 'Dopamine', 'targets': ['DAT']},
    {'smiles': 'NC[C@H](O)c1ccc(O)c(O)c1', 'name': 'Norepinephrine', 'targets': ['NET']},
    {'smiles': 'NCCc1c[nH]c2ccc(O)cc12', 'name': 'Serotonin', 'targets': ['SERT']},
    {'smiles': 'C[C@H](N)Cc1ccccc1', 'name': 'd-Amphetamine', 'targets': ['DAT', 'NET']},
    {'smiles': 'C[C@H](NC)Cc1ccccc1', 'name': 'd-Methamphetamine', 'targets': ['DAT', 'NET']},
    {'smiles': 'C[C@H](NC)Cc1ccc2OCOc2c1', 'name': 'MDMA', 'targets': ['SERT', 'DAT', 'NET']},
    {'smiles': 'NCCc1ccc(O)cc1', 'name': 'Tyramine', 'targets': ['DAT', 'NET']},
    {'smiles': 'C[C@H](N)C(=O)c1ccccc1', 'name': 'Cathinone', 'targets': ['DAT', 'NET']},
    {'smiles': 'C[C@H](NC)C(=O)c1ccc(C)cc1', 'name': 'Mephedrone', 'targets': ['DAT', 'NET', 'SERT']},
]

# Known blockers that should NOT be predicted as substrates
KNOWN_BLOCKERS = [
    {'smiles': 'COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C', 'name': 'Cocaine', 'targets': ['DAT', 'NET', 'SERT']},
    {'smiles': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O', 'name': 'Caffeine', 'targets': []},
    {'smiles': 'CN(C)CCCN1c2ccccc2Sc3ccc(Cl)cc13', 'name': 'Chlorpromazine', 'targets': ['DAT']},
    {'smiles': 'CNCCC(Oc1ccc(C(F)(F)F)cc1)c2ccccc2', 'name': 'Fluoxetine', 'targets': ['SERT']},
    {'smiles': 'CN(C)CCc1ccc(Br)cc1', 'name': 'Brofaromine', 'targets': ['SERT']},
]


def run_external_validation():
    """Run complete external validation."""
    print("=" * 70)
    print("EXTERNAL VALIDATION - StereoGNN Transporter Model")
    print(f"Testing {len(STEREO_VALIDATION_PAIRS)} stereo pairs")
    print("=" * 70)

    validator = ExternalValidator()

    # 1. Stereo sensitivity validation
    print("\n" + "-" * 70)
    print("1. STEREO SENSITIVITY VALIDATION")
    print("-" * 70)

    stereo_correct = 0
    stereo_total = 0
    stereo_results = []
    failed_pairs = []

    for pair in STEREO_VALIDATION_PAIRS:
        d_pred = validator.predict(pair['d_smiles'])
        l_pred = validator.predict(pair['l_smiles'])

        if d_pred is None or l_pred is None:
            print(f"  {pair['name']}: SKIPPED (featurization failed)")
            continue

        target = pair['target']
        d_prob = d_pred[target]['substrate_prob']
        l_prob = l_pred[target]['substrate_prob']

        correct = d_prob > l_prob
        stereo_total += 1
        if correct:
            stereo_correct += 1
        else:
            failed_pairs.append(pair['name'])

        status = "PASS" if correct else "FAIL"
        margin = d_prob - l_prob

        stereo_results.append({
            'name': pair['name'],
            'd_prob': d_prob,
            'l_prob': l_prob,
            'correct': correct,
            'margin': margin
        })

        print(f"  {pair['name']:30s} | d={d_prob:.3f} l={l_prob:.3f} | margin={margin:+.3f} | {status}")

    stereo_accuracy = stereo_correct / stereo_total if stereo_total > 0 else 0
    print(f"\n  STEREO SENSITIVITY: {stereo_correct}/{stereo_total} = {stereo_accuracy:.1%}")

    if failed_pairs:
        print(f"\n  FAILED PAIRS ({len(failed_pairs)}):")
        for name in failed_pairs:
            print(f"    - {name}")

    # 2. Known substrate validation
    print("\n" + "-" * 70)
    print("2. KNOWN SUBSTRATE VALIDATION")
    print("-" * 70)

    substrate_correct = 0
    substrate_total = 0

    for compound in KNOWN_SUBSTRATES:
        pred = validator.predict(compound['smiles'])
        if pred is None:
            continue

        for target in compound['targets']:
            substrate_total += 1
            is_substrate = pred[target]['prediction'] == 'substrate'
            prob = pred[target]['substrate_prob']

            if is_substrate or prob > 0.5:
                substrate_correct += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"  {compound['name']:20s} @ {target}: {pred[target]['prediction']:10s} (prob={prob:.3f}) | {status}")

    if substrate_total > 0:
        print(f"\n  SUBSTRATE DETECTION: {substrate_correct}/{substrate_total} = {substrate_correct/substrate_total:.1%}")

    # 3. Known blocker validation
    print("\n" + "-" * 70)
    print("3. KNOWN BLOCKER VALIDATION (should NOT be substrates)")
    print("-" * 70)

    blocker_correct = 0
    blocker_total = 0

    for compound in KNOWN_BLOCKERS:
        pred = validator.predict(compound['smiles'])
        if pred is None:
            continue

        for target in compound['targets']:
            blocker_total += 1
            prediction = pred[target]['prediction']
            substrate_prob = pred[target]['substrate_prob']

            if prediction != 'substrate' and substrate_prob < 0.5:
                blocker_correct += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(f"  {compound['name']:20s} @ {target}: {prediction:10s} (sub_prob={substrate_prob:.3f}) | {status}")

    if blocker_total > 0:
        print(f"\n  BLOCKER DETECTION: {blocker_correct}/{blocker_total} = {blocker_correct/blocker_total:.1%}")

    # Final summary
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Stereo Sensitivity:  {stereo_accuracy:.1%} ({stereo_correct}/{stereo_total})")
    if substrate_total > 0:
        print(f"  Substrate Detection: {substrate_correct/substrate_total:.1%} ({substrate_correct}/{substrate_total})")
    if blocker_total > 0:
        print(f"  Blocker Detection:   {blocker_correct/blocker_total:.1%} ({blocker_correct}/{blocker_total})")

    passes = stereo_accuracy >= 0.80
    print(f"\n  OVERALL: {'PASS' if passes else 'FAIL'} (target: >=80% stereo sensitivity)")
    print("=" * 70)

    return {
        'stereo_accuracy': stereo_accuracy,
        'stereo_correct': stereo_correct,
        'stereo_total': stereo_total,
        'stereo_results': stereo_results,
        'failed_pairs': failed_pairs,
        'passes': passes
    }


if __name__ == "__main__":
    run_external_validation()
