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
import logging
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

logger = logging.getLogger("stereognn")


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the pipeline."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("stereognn.log", mode="a"),
        ],
    )


def train(args: argparse.Namespace) -> None:
    """Train the StereoGNN model."""
    logger.info("=" * 60)
    logger.info("TRAINING STEREOGNN")
    logger.info("=" * 60)

    # Step 1: Data curation
    logger.info("[1/4] Curating data...")
    pipeline = DataCurationPipeline()
    splits = pipeline.run(use_cache=args.use_cache)

    for name, df in splits.items():
        stats = pipeline.get_statistics(df)
        logger.info("  %s: %d compounds", name, stats["total_compounds"])
        for target in ["DAT", "NET", "SERT"]:
            logger.info(
                "    %s: %d substrates, %d blockers",
                target,
                stats[f"{target}_substrates"],
                stats[f"{target}_blockers"],
            )

    # Step 2: Create dataloaders
    logger.info("[2/4] Creating dataloaders...")
    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3d=False,
    )

    if len(dataloaders["train"].dataset) == 0:
        logger.error("No training data available. Ensure ChEMBL data is accessible.")
        return

    # Step 3: Initialize model
    logger.info("[3/4] Initializing model...")
    model = StereoGNN()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Model parameters: %s", f"{n_params:,}")

    # Step 4: Train
    logger.info("[4/4] Training...")
    trainer = Trainer(
        model=model,
        experiment_name=args.experiment_name,
    )
    trainer.setup(dataloaders["train"].dataset)

    history = trainer.train(
        dataloaders["train"],
        dataloaders["val"],
        num_epochs=args.epochs,
    )

    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(dataloaders["test"])
    evaluator.print_results(results)

    # Save results
    results_path = CONFIG.data.results_dir / f"{args.experiment_name}_results.json"
    evaluator.save_results(results, results_path)
    logger.info("Results saved to %s", results_path)

    # Check success criteria
    if results.passes_criteria:
        logger.info("SUCCESS: ALL CRITERIA PASSED")
    else:
        logger.warning("FAILED: Some criteria not met")
        for criterion in results.failed_criteria:
            logger.warning("  - %s", criterion)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model."""
    logger.info("EVALUATING MODEL")

    model_path = CONFIG.data.models_dir / args.experiment_name / "best_model.pt"
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return

    dataloaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3d=False,
    )

    results = evaluate_model(
        model_path=model_path,
        test_loader=dataloaders["test"],
    )
    return results


def ablation(args: argparse.Namespace) -> None:
    """Run ablation study."""
    logger.info("RUNNING ABLATION STUDY")

    pipeline = DataCurationPipeline()
    pipeline.run(use_cache=True)

    study = run_ablation_study(num_epochs=args.epochs)

    stereo_abl = next(
        (a for a in study.ablations if a.name == "no_stereochemistry"),
        None,
    )
    if stereo_abl:
        drop = -stereo_abl.auroc_delta
        logger.info("Stereo feature ablation drop = %.4f", drop)
        if drop >= 0.05:
            logger.info("PASS: Stereochemistry features contribute >= 5%% AUROC")
        else:
            logger.warning("WARN: Stereochemistry contribution < 5%%")


def predict(args: argparse.Namespace) -> None:
    """Predict a single molecule."""
    logger.info("SINGLE MOLECULE PREDICTION")

    model_path = CONFIG.data.models_dir / args.experiment_name / "best_model.pt"
    result = predict_single(args.smiles, model_path)

    logger.info("Molecule: %s", result["smiles"])
    logger.info("Valid: %s", result["is_valid"])

    if result["is_valid"]:
        logger.info("Predictions:")
        logger.info("  DAT: %-10s (substrate prob: %.3f)", result["dat_prediction"], result["dat_substrate_prob"])
        logger.info("  NET: %-10s (substrate prob: %.3f)", result["net_prediction"], result["net_substrate_prob"])
        logger.info("  SERT: %-10s (substrate prob: %.3f)", result["sert_prediction"], result["sert_substrate_prob"])
        logger.info("Applicability Domain: in_domain=%s (score: %.3f)", result["in_domain"], result["domain_score"])
        logger.info("Stereocenters: %d", result["num_stereocenters"])


def screen(args: argparse.Namespace) -> None:
    """Run virtual screening."""
    logger.info("VIRTUAL SCREENING")

    model_path = CONFIG.data.models_dir / args.experiment_name / "best_model.pt"
    predictor = TransporterPredictor(model_path=model_path)

    with open(args.input_file) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    logger.info("Loaded %d molecules", len(smiles_list))

    results = predictor.virtual_screen(
        smiles_list,
        target=args.target,
        min_substrate_prob=args.threshold,
    )

    results.to_csv(args.output_file, index=False)
    logger.info("Results saved to %s", args.output_file)

    logger.info("Top 10 hits for %s:", args.target)
    for _, row in results.head(10).iterrows():
        logger.info("  %3d. %-40s prob=%.3f", row["rank"], row["smiles"][:40], row["substrate_prob"])


def main() -> None:
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

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "ablation", "predict", "screen"],
                        help="Mode to run")
    parser.add_argument("--experiment_name", type=str, default="stereo_gnn_v1",
                        help="Experiment name for saving/loading")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use cached data if available")
    parser.add_argument("--smiles", type=str,
                        help="SMILES string for prediction")
    parser.add_argument("--input_file", type=str,
                        help="Input file for screening")
    parser.add_argument("--output_file", type=str, default="screening_results.csv",
                        help="Output file for screening")
    parser.add_argument("--target", type=str, default="DAT",
                        choices=["DAT", "NET", "SERT"],
                        help="Target transporter for screening")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Minimum substrate probability threshold")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity level")

    args = parser.parse_args()

    setup_logging(args.log_level)
    torch.set_num_threads(args.num_workers)

    mode_map = {
        "train": train,
        "evaluate": evaluate,
        "ablation": ablation,
        "predict": predict,
        "screen": screen,
    }

    if args.mode == "predict" and not args.smiles:
        logger.error("--smiles required for predict mode")
        sys.exit(1)
    if args.mode == "screen" and not args.input_file:
        logger.error("--input_file required for screen mode")
        sys.exit(1)

    mode_map[args.mode](args)


if __name__ == "__main__":
    main()
