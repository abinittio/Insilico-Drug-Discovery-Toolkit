"""
StereoGNN Transporter Substrate Predictor
==========================================

Main entry point for the complete pipeline.

Usage:
    python main.py --mode train           # Train the model
    python main.py --mode evaluate        # Evaluate trained model
    python main.py --mode ablation        # Run ablation study
    python main.py --mode predict         # Predict a single molecule
    python main.py --mode screen          # Virtual screening

Success Criteria (MUST ACHIEVE):
- Overall ROC-AUC: >= 0.85
- Monoamine-specific ROC-AUC: >= 0.95
- PR-AUC: >= 0.65
- Stereo selectivity: >= 80% correct on known pairs
- Ablation drop (no stereo): >= 5%
"""

import argparse
import sys
from pathlib import Path

import torch

from config import CONFIG
from data_curation import DataCurationPipeline
from dataset import create_dataloaders
from model import StereoGNN
from trainer import Trainer
from evaluation import ModelEvaluator, evaluate_model
from ablation import run_ablation_study
from inference import TransporterPredictor, predict_single


def train(args):
    """Train the StereoGNN model."""
    print("=" * 70)
    print("TRAINING STEREOGNN")
    print("=" * 70)

    # Step 1: Data curation
    print("\n[1/4] Curating data...")
    pipeline = DataCurationPipeline()
    splits = pipeline.run(use_cache=args.use_cache)

    for name, df in splits.items():
        stats = pipeline.get_statistics(df)
        print(f"  {name}: {stats['total_compounds']} compounds")
        for target in ['DAT', 'NET', 'SERT']:
            print(f"    {target}: {stats[f'{target}_substrates']} substrates, "
                  f"{stats[f'{target}_blockers']} blockers")

    # Step 2: Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3d=False,
    )

    if len(dataloaders['train'].dataset) == 0:
        print("ERROR: No training data available!")
        print("Please ensure ChEMBL data is accessible or add more literature data.")
        return

    # Step 3: Initialize model
    print("\n[3/4] Initializing model...")
    model = StereoGNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Step 4: Train
    print("\n[4/4] Training...")
    trainer = Trainer(
        model=model,
        experiment_name=args.experiment_name,
    )
    trainer.setup(dataloaders['train'].dataset)

    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=args.epochs,
    )

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(dataloaders['test'])
    evaluator.print_results(results)

    # Save results
    results_path = CONFIG.data.results_dir / f"{args.experiment_name}_results.json"
    evaluator.save_results(results, results_path)
    print(f"\nResults saved to {results_path}")

    # Check success criteria
    print("\n" + "=" * 70)
    if results.passes_criteria:
        print("SUCCESS: ALL CRITERIA PASSED")
    else:
        print("FAILED: Some criteria not met")
        for criterion in results.failed_criteria:
            print(f"  - {criterion}")
    print("=" * 70)


def evaluate(args):
    """Evaluate a trained model."""
    print("=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)

    model_path = CONFIG.data.models_dir / args.experiment_name / "best_model.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    # Create test dataloader
    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3d=False,
    )

    results = evaluate_model(
        model_path=model_path,
        test_loader=dataloaders['test'],
    )

    return results


def ablation(args):
    """Run ablation study."""
    print("=" * 70)
    print("RUNNING ABLATION STUDY")
    print("=" * 70)

    # First ensure data is curated
    pipeline = DataCurationPipeline()
    pipeline.run(use_cache=True)

    study = run_ablation_study(num_epochs=args.epochs)

    # Check stereo ablation result
    stereo_abl = next(
        (a for a in study.ablations if a.name == "no_stereochemistry"),
        None
    )
    if stereo_abl:
        drop = -stereo_abl.auroc_delta
        print(f"\n" + "=" * 70)
        print(f"CRITICAL: Stereo feature ablation drop = {drop:.4f}")
        if drop >= 0.05:
            print("PASS: Stereochemistry features contribute >= 5% AUROC")
        else:
            print("WARN: Stereochemistry contribution < 5%")
        print("=" * 70)


def predict(args):
    """Predict a single molecule."""
    print("=" * 70)
    print("SINGLE MOLECULE PREDICTION")
    print("=" * 70)

    model_path = CONFIG.data.models_dir / args.experiment_name / "best_model.pt"

    result = predict_single(args.smiles, model_path)

    print(f"\nMolecule: {result['smiles']}")
    print(f"Valid: {result['is_valid']}")

    if result['is_valid']:
        print(f"\nPredictions:")
        print(f"  DAT: {result['dat_prediction']:<10} (substrate prob: {result['dat_substrate_prob']:.3f})")
        print(f"  NET: {result['net_prediction']:<10} (substrate prob: {result['net_substrate_prob']:.3f})")
        print(f"  SERT: {result['sert_prediction']:<10} (substrate prob: {result['sert_substrate_prob']:.3f})")

        print(f"\nApplicability Domain:")
        print(f"  In domain: {result['in_domain']} (score: {result['domain_score']:.3f})")

        print(f"\nStereochemistry:")
        print(f"  Has stereocenters: {result['has_stereocenters']}")
        print(f"  Number of stereocenters: {result['num_stereocenters']}")


def screen(args):
    """Run virtual screening."""
    print("=" * 70)
    print("VIRTUAL SCREENING")
    print("=" * 70)

    model_path = CONFIG.data.models_dir / args.experiment_name / "best_model.pt"

    predictor = TransporterPredictor(model_path=model_path)

    # Read SMILES from file
    with open(args.input_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smiles_list)} molecules")

    results = predictor.virtual_screen(
        smiles_list,
        target=args.target,
        min_substrate_prob=args.threshold,
    )

    # Save results
    results.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

    # Print top hits
    print(f"\nTop 10 hits for {args.target}:")
    for i, row in results.head(10).iterrows():
        print(f"  {row['rank']:3d}. {row['smiles'][:40]:<40} prob={row['substrate_prob']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="StereoGNN Transporter Substrate Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train --epochs 100
  python main.py --mode evaluate
  python main.py --mode predict --smiles "C[C@H](N)Cc1ccccc1"
  python main.py --mode screen --input molecules.txt --target DAT
        """,
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'ablation', 'predict', 'screen'],
        help='Mode to run',
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='stereo_gnn_v1',
        help='Experiment name for saving/loading',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers',
    )
    parser.add_argument(
        '--use_cache',
        action='store_true',
        help='Use cached data if available',
    )
    parser.add_argument(
        '--smiles',
        type=str,
        help='SMILES string for prediction',
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='Input file for screening',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='screening_results.csv',
        help='Output file for screening',
    )
    parser.add_argument(
        '--target',
        type=str,
        default='DAT',
        choices=['DAT', 'NET', 'SERT'],
        help='Target transporter for screening',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum substrate probability threshold',
    )

    args = parser.parse_args()

    # Set number of threads for PyTorch
    torch.set_num_threads(args.num_workers)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'ablation':
        ablation(args)
    elif args.mode == 'predict':
        if not args.smiles:
            print("Error: --smiles required for predict mode")
            sys.exit(1)
        predict(args)
    elif args.mode == 'screen':
        if not args.input_file:
            print("Error: --input_file required for screen mode")
            sys.exit(1)
        screen(args)


if __name__ == "__main__":
    main()
