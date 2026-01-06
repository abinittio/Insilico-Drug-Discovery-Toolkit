"""
Kinetic Pipeline Sanity Check
=============================

Verifies the complete kinetic extension works end-to-end:
1. Dataset loading with kinetic labels
2. Model instantiation (StereoGNNKinetic)
3. Forward pass and loss computation
4. Inference API (KineticTransporterPredictor)
5. Uncertainty quantification

Run with: python test_kinetic_pipeline.py
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_success(msg: str):
    print(f"  [OK] {msg}")


def print_error(msg: str):
    print(f"  [FAIL] {msg}")


def print_info(msg: str):
    print(f"  [INFO] {msg}")


def create_test_kinetic_data() -> pd.DataFrame:
    """Create a small test dataset with kinetic labels."""
    # Test compounds with known properties
    data = [
        # Substrates
        {"smiles": "C[C@H](N)Cc1ccccc1", "target": "DAT", "label": 2,
         "pKi": 7.2, "pIC50": 6.8, "interaction_mode": 0, "kinetic_bias": 0.75, "confidence": 1.0},
        {"smiles": "C[C@H](N)Cc1ccccc1", "target": "NET", "label": 2,
         "pKi": 7.5, "pIC50": 7.0, "interaction_mode": 0, "kinetic_bias": 0.70, "confidence": 1.0},
        {"smiles": "C[C@H](N)Cc1ccccc1", "target": "SERT", "label": 1,
         "pKi": 5.5, "pIC50": 5.0, "interaction_mode": 1, "kinetic_bias": 0.3, "confidence": 1.0},

        # l-Amphetamine (different stereo)
        {"smiles": "C[C@@H](N)Cc1ccccc1", "target": "DAT", "label": 1,
         "pKi": 5.8, "pIC50": 5.5, "interaction_mode": 3, "kinetic_bias": 0.4, "confidence": 1.0},
        {"smiles": "C[C@@H](N)Cc1ccccc1", "target": "NET", "label": 1,
         "pKi": 6.0, "pIC50": 5.7, "interaction_mode": 3, "kinetic_bias": 0.4, "confidence": 1.0},
        {"smiles": "C[C@@H](N)Cc1ccccc1", "target": "SERT", "label": 0,
         "pKi": float('nan'), "pIC50": float('nan'), "interaction_mode": -1, "kinetic_bias": float('nan'), "confidence": 0.8},

        # Cocaine (blocker)
        {"smiles": "COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3", "target": "DAT", "label": 1,
         "pKi": 7.0, "pIC50": 6.5, "interaction_mode": 1, "kinetic_bias": 0.0, "confidence": 1.0},
        {"smiles": "COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3", "target": "NET", "label": 1,
         "pKi": 6.5, "pIC50": 6.0, "interaction_mode": 1, "kinetic_bias": 0.0, "confidence": 1.0},
        {"smiles": "COC(=O)C1CC2CCC(C1)N2C(=O)c3ccccc3", "target": "SERT", "label": 1,
         "pKi": 6.8, "pIC50": 6.3, "interaction_mode": 1, "kinetic_bias": 0.0, "confidence": 1.0},

        # Dopamine (endogenous substrate)
        {"smiles": "NCCc1ccc(O)c(O)c1", "target": "DAT", "label": 2,
         "pKi": 6.0, "pIC50": 5.5, "interaction_mode": 0, "kinetic_bias": 0.9, "confidence": 1.0},
        {"smiles": "NCCc1ccc(O)c(O)c1", "target": "NET", "label": 2,
         "pKi": 6.2, "pIC50": 5.8, "interaction_mode": 0, "kinetic_bias": 0.85, "confidence": 1.0},
        {"smiles": "NCCc1ccc(O)c(O)c1", "target": "SERT", "label": 0,
         "pKi": float('nan'), "pIC50": float('nan'), "interaction_mode": -1, "kinetic_bias": float('nan'), "confidence": 0.7},

        # Caffeine (inactive)
        {"smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "target": "DAT", "label": 0,
         "pKi": float('nan'), "pIC50": float('nan'), "interaction_mode": -1, "kinetic_bias": float('nan'), "confidence": 0.9},
        {"smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "target": "NET", "label": 0,
         "pKi": float('nan'), "pIC50": float('nan'), "interaction_mode": -1, "kinetic_bias": float('nan'), "confidence": 0.9},
        {"smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "target": "SERT", "label": 0,
         "pKi": float('nan'), "pIC50": float('nan'), "interaction_mode": -1, "kinetic_bias": float('nan'), "confidence": 0.9},

        # MDMA (substrate)
        {"smiles": "C[C@H](NC)Cc1ccc2OCOc2c1", "target": "DAT", "label": 2,
         "pKi": 6.5, "pIC50": 6.2, "interaction_mode": 0, "kinetic_bias": 0.55, "confidence": 1.0},
        {"smiles": "C[C@H](NC)Cc1ccc2OCOc2c1", "target": "NET", "label": 2,
         "pKi": 6.8, "pIC50": 6.5, "interaction_mode": 0, "kinetic_bias": 0.55, "confidence": 1.0},
        {"smiles": "C[C@H](NC)Cc1ccc2OCOc2c1", "target": "SERT", "label": 2,
         "pKi": 7.5, "pIC50": 7.2, "interaction_mode": 0, "kinetic_bias": 0.40, "confidence": 1.0},
    ]

    return pd.DataFrame(data)


def test_dataset_loading():
    """Test 1: Verify KineticTransporterDataset loads correctly."""
    print_section("Test 1: Dataset Loading")

    try:
        from dataset import KineticTransporterDataset, batch_to_kinetic_targets
        from featurizer import MoleculeGraphFeaturizer
        from config import CONFIG
        from torch_geometric.loader import DataLoader as PyGDataLoader

        # Create test data
        test_df = create_test_kinetic_data()
        data_dir = CONFIG.data.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save as train split
        test_df.to_parquet(data_dir / "train.parquet")
        test_df.to_parquet(data_dir / "val.parquet")
        test_df.to_parquet(data_dir / "test.parquet")
        print_success(f"Created test data at {data_dir}")

        # Create featurizer
        featurizer = MoleculeGraphFeaturizer(use_3d=False)
        print_success("MoleculeGraphFeaturizer created")

        # Create dataset
        dataset = KineticTransporterDataset(
            data_path=data_dir,
            split='train',
            featurizer=featurizer,
            pre_featurize=True,
            use_3d=False,
        )
        print_success(f"KineticTransporterDataset created with {len(dataset)} samples")

        # Check a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print_info(f"Sample node features shape: {sample.x.shape}")
            print_info(f"Sample edge features shape: {sample.edge_attr.shape}")

            # Check kinetic labels exist
            has_kinetic = hasattr(sample, 'y_dat_pki')
            print_success(f"Kinetic labels present: {has_kinetic}")

            if has_kinetic:
                print_info(f"  y_dat_pki: {sample.y_dat_pki}")
                print_info(f"  y_dat_mode: {sample.y_dat_mode}")

        # Test dataloader
        loader = PyGDataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print_success(f"DataLoader works, batch size: {batch.num_graphs}")

        # Test batch_to_kinetic_targets
        targets = batch_to_kinetic_targets(batch)
        print_success(f"batch_to_kinetic_targets works, keys: {list(targets.keys())[:5]}...")

        # Get statistics
        stats = dataset.get_kinetic_statistics()
        print_info(f"Dataset stats: {len(stats)} metrics")

        return True, dataset, loader

    except Exception as e:
        print_error(f"Dataset loading failed: {e}")
        traceback.print_exc()
        return False, None, None


def test_model_instantiation():
    """Test 2: Verify StereoGNNKinetic instantiates correctly."""
    print_section("Test 2: Model Instantiation")

    try:
        from model import StereoGNN, StereoGNNKinetic, KineticHead, count_parameters

        # Test KineticHead
        head = KineticHead(input_dim=384, hidden_dim=128)
        print_success(f"KineticHead created, params: {count_parameters(head):,}")

        # Test input/output
        test_input = torch.randn(4, 384)
        output = head(test_input)
        print_success(f"KineticHead forward pass works")
        print_info(f"  Output keys: {list(output.keys())}")
        print_info(f"  pKi_mean shape: {output['pKi_mean'].shape}")
        print_info(f"  interaction_mode shape: {output['interaction_mode'].shape}")

        # Test StereoGNNKinetic
        model = StereoGNNKinetic()
        print_success(f"StereoGNNKinetic created, params: {count_parameters(model):,}")

        # Test from_pretrained
        base_model = StereoGNN()
        kinetic_from_pretrained = StereoGNNKinetic.from_pretrained(base_model, freeze_backbone=False)
        print_success("StereoGNNKinetic.from_pretrained works")

        return True, model

    except Exception as e:
        print_error(f"Model instantiation failed: {e}")
        traceback.print_exc()
        return False, None


def test_forward_pass(model, loader):
    """Test 3: Verify forward pass and loss computation."""
    print_section("Test 3: Forward Pass & Loss")

    try:
        from losses import KineticMultiTaskLoss
        from dataset import batch_to_kinetic_targets

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_info(f"Using device: {device}")

        model = model.to(device)
        model.eval()

        # Get a batch
        batch = next(iter(loader)).to(device)
        print_success(f"Batch loaded to {device}")

        # Forward pass
        with torch.no_grad():
            output = model(batch, return_kinetics=True, return_attention=True)

        print_success("Forward pass completed")
        print_info(f"Output keys: {list(output.keys())[:10]}...")

        # Check outputs
        assert 'DAT' in output, "Missing DAT classification output"
        assert 'DAT_pKi_mean' in output, "Missing DAT_pKi_mean"
        assert 'DAT_interaction_mode' in output, "Missing DAT_interaction_mode"
        print_success("All expected output keys present")

        print_info(f"DAT logits shape: {output['DAT'].shape}")
        print_info(f"DAT_pKi_mean: {output['DAT_pKi_mean']}")
        print_info(f"DAT_kinetic_bias_mean: {output['DAT_kinetic_bias_mean']}")

        # Test loss computation
        criterion = KineticMultiTaskLoss(learn_weights=True).to(device)
        targets = batch_to_kinetic_targets(batch)

        losses = criterion(output, targets)
        print_success(f"Loss computation works, total loss: {losses['total'].item():.4f}")
        print_info(f"Individual losses: {[(k, v.item()) for k, v in losses.items() if k != 'total'][:5]}")

        # Check task weights
        weights = criterion.get_task_weights()
        print_info(f"Task weights: {weights}")

        return True

    except Exception as e:
        print_error(f"Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_uncertainty_quantification(model, loader):
    """Test 4: Verify uncertainty quantification works."""
    print_section("Test 4: Uncertainty Quantification")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        batch = next(iter(loader)).to(device)

        # Test predict_kinetics_with_uncertainty
        print_info("Running MC Dropout (10 samples)...")
        results = model.predict_kinetics_with_uncertainty(batch, n_mc_samples=10)

        print_success("predict_kinetics_with_uncertainty completed")

        # Check structure
        assert 'DAT' in results, "Missing DAT in results"
        assert 'pKi' in results['DAT'], "Missing pKi in DAT results"
        assert 'mean' in results['DAT']['pKi'], "Missing mean in pKi results"
        assert 'aleatoric' in results['DAT']['pKi'], "Missing aleatoric uncertainty"
        assert 'epistemic' in results['DAT']['pKi'], "Missing epistemic uncertainty"
        assert 'total' in results['DAT']['pKi'], "Missing total uncertainty"

        print_success("All uncertainty components present")

        # Print sample results
        for task in ['DAT', 'NET', 'SERT']:
            pki = results[task]['pKi']
            print_info(f"{task} pKi: {pki['mean'][0].item():.2f} ± {pki['total'][0].item():.2f}")
            print_info(f"  (aleatoric: {pki['aleatoric'][0].item():.2f}, epistemic: {pki['epistemic'][0].item():.2f})")

        # Check mode classification
        mode = results['DAT']['interaction_mode']
        print_info(f"DAT mode probs: {mode['mean_probs'][0].tolist()}")
        print_info(f"DAT predicted mode: {mode['class_names'][mode['predicted_class'][0].item()]}")

        return True

    except Exception as e:
        print_error(f"Uncertainty quantification failed: {e}")
        traceback.print_exc()
        return False


def test_inference_api():
    """Test 5: Verify KineticTransporterPredictor works."""
    print_section("Test 5: Inference API")

    try:
        from inference import KineticTransporterPredictor, KineticPrediction

        # Create predictor (no trained weights)
        predictor = KineticTransporterPredictor(n_mc_samples=10)
        print_success("KineticTransporterPredictor created")

        # Test prediction
        test_smiles = [
            "C[C@H](N)Cc1ccccc1",  # d-Amphetamine
            "C[C@@H](N)Cc1ccccc1",  # l-Amphetamine
            "NCCc1ccc(O)c(O)c1",    # Dopamine
        ]

        for smi in test_smiles:
            result = predictor.predict(smi)
            print_success(f"Predicted: {smi[:30]}...")

            if result.is_valid and result.dat_kinetics:
                k = result.dat_kinetics
                print_info(f"  DAT: mode={k.interaction_mode}, pKi={k.pKi:.2f}±{k.pKi_uncertainty:.2f}")
                print_info(f"  {k.mechanism_summary}")

        # Test batch prediction
        batch_results = predictor.predict_batch(test_smiles, show_progress=False)
        print_success(f"Batch prediction works, got {len(batch_results)} results")

        # Test mechanism comparison
        comparison = predictor.compare_mechanisms(
            "C[C@H](N)Cc1ccccc1",   # d-Amph
            "C[C@@H](N)Cc1ccccc1",  # l-Amph
            target="DAT"
        )
        print_success("Enantiomer comparison works")
        print_info(f"  pKi difference: {comparison['pKi_diff']:.2f}")
        print_info(f"  Same mode: {comparison['same_mode']}")

        # Test get_summary
        result = predictor.predict("C[C@H](N)Cc1ccccc1")
        summary = result.get_summary()
        print_success("get_summary() works")
        print_info(f"Summary preview:\n{summary[:200]}...")

        return True

    except Exception as e:
        print_error(f"Inference API failed: {e}")
        traceback.print_exc()
        return False


def test_training_script_imports():
    """Test 6: Verify training script can be imported."""
    print_section("Test 6: Training Script Imports")

    try:
        from run_training_kinetic import (
            KineticTrainer,
            CosineWarmupScheduler,
            EarlyStopping,
        )
        print_success("Training script imports work")

        # Test scheduler
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=100)
        print_success("CosineWarmupScheduler instantiated")

        # Test early stopping
        es = EarlyStopping(patience=10)
        es(0.5)
        es(0.6)
        print_success(f"EarlyStopping works, best: {es.best_score}")

        return True

    except Exception as e:
        print_error(f"Training script import failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" KINETIC PIPELINE SANITY CHECK")
    print("="*60)

    results = {}

    # Test 1: Dataset
    success, dataset, loader = test_dataset_loading()
    results['dataset'] = success

    if not success:
        print("\nDataset loading failed, cannot continue with other tests.")
        return results

    # Test 2: Model
    success, model = test_model_instantiation()
    results['model'] = success

    if not success:
        print("\nModel instantiation failed, cannot continue.")
        return results

    # Test 3: Forward pass
    results['forward'] = test_forward_pass(model, loader)

    # Test 4: Uncertainty
    results['uncertainty'] = test_uncertainty_quantification(model, loader)

    # Test 5: Inference
    results['inference'] = test_inference_api()

    # Test 6: Training script
    results['training'] = test_training_script_imports()

    # Summary
    print_section("TEST SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n" + "="*60)
        print(" ALL TESTS PASSED - PIPELINE IS READY FOR TRAINING!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run data curation: python data_curation_kinetic.py")
        print("  2. Start training: python run_training_kinetic.py --epochs 100")
        print("  3. (Optional) Use Gradio interface for demo")
    else:
        print("\n" + "="*60)
        print(" SOME TESTS FAILED - PLEASE FIX BEFORE TRAINING")
        print("="*60)

    return results


if __name__ == "__main__":
    main()
