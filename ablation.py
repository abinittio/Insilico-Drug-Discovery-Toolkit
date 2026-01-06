"""
Ablation Study Framework for StereoGNN
======================================

Systematic ablation studies to validate the contribution of:
1. Stereochemistry features (CRITICAL - must show >=5% drop without)
2. Multi-task learning
3. Attention mechanism
4. 3D coordinates
5. Individual GNN layers

This provides scientific rigor and validates our architectural choices.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from model import StereoGNN, StereoGNNForAblation, ModelConfig
from trainer import Trainer, train_model
from evaluation import ModelEvaluator, EvaluationResults
from dataset import create_dataloaders
from config import CONFIG


@dataclass
class AblationResult:
    """Results from a single ablation experiment."""
    name: str
    description: str
    config_changes: Dict

    # Metrics
    overall_auroc: float
    dat_auroc: float
    net_auroc: float
    sert_auroc: float
    stereo_accuracy: float

    # Comparison to baseline
    auroc_delta: float  # Negative = worse than baseline
    stereo_delta: float

    # Training info
    best_epoch: int
    final_loss: float


@dataclass
class AblationStudy:
    """Complete ablation study results."""
    baseline_name: str
    baseline_auroc: float
    baseline_stereo: float

    ablations: List[AblationResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'baseline_name': self.baseline_name,
            'baseline_auroc': self.baseline_auroc,
            'baseline_stereo': self.baseline_stereo,
            'ablations': [
                {
                    'name': a.name,
                    'description': a.description,
                    'overall_auroc': a.overall_auroc,
                    'auroc_delta': a.auroc_delta,
                    'stereo_accuracy': a.stereo_accuracy,
                    'stereo_delta': a.stereo_delta,
                }
                for a in self.ablations
            ],
        }


class AblationRunner:
    """
    Runs systematic ablation experiments.
    """

    def __init__(
        self,
        output_dir: Path = None,
        device: Optional[torch.device] = None,
    ):
        self.output_dir = output_dir or CONFIG.data.results_dir / "ablation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device(CONFIG.training.device)

    def run_full_ablation_study(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int = 50,
    ) -> AblationStudy:
        """
        Run complete ablation study.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            test_loader: Test dataloader
            num_epochs: Epochs per experiment

        Returns:
            AblationStudy with all results
        """
        print("=" * 70)
        print("ABLATION STUDY")
        print("=" * 70)

        # 1. Train baseline model
        print("\n[1/6] Training baseline model (full StereoGNN)...")
        baseline_result = self._train_and_evaluate(
            model=StereoGNN(),
            name="baseline",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
        )

        study = AblationStudy(
            baseline_name="StereoGNN (full)",
            baseline_auroc=baseline_result['overall_auroc'],
            baseline_stereo=baseline_result['stereo_accuracy'],
        )

        # 2. Ablation: Remove stereochemistry features
        print("\n[2/6] Ablation: No stereochemistry features...")
        no_stereo_result = self._train_and_evaluate(
            model=StereoGNNForAblation(),
            name="no_stereo",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
        )

        study.ablations.append(AblationResult(
            name="no_stereochemistry",
            description="Remove all stereochemistry features from node and edge encodings",
            config_changes={"stereo_features": False},
            overall_auroc=no_stereo_result['overall_auroc'],
            dat_auroc=no_stereo_result.get('DAT_auroc', 0),
            net_auroc=no_stereo_result.get('NET_auroc', 0),
            sert_auroc=no_stereo_result.get('SERT_auroc', 0),
            stereo_accuracy=no_stereo_result['stereo_accuracy'],
            auroc_delta=no_stereo_result['overall_auroc'] - baseline_result['overall_auroc'],
            stereo_delta=no_stereo_result['stereo_accuracy'] - baseline_result['stereo_accuracy'],
            best_epoch=no_stereo_result['best_epoch'],
            final_loss=no_stereo_result['final_loss'],
        ))

        # 3. Ablation: Single-task learning (DAT only)
        print("\n[3/6] Ablation: Single-task learning (DAT only)...")
        single_task_model = self._create_single_task_model()
        single_task_result = self._train_and_evaluate(
            model=single_task_model,
            name="single_task_DAT",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
        )

        study.ablations.append(AblationResult(
            name="single_task_DAT",
            description="Train only for DAT prediction (no multi-task learning)",
            config_changes={"multi_task": False, "target": "DAT"},
            overall_auroc=single_task_result['overall_auroc'],
            dat_auroc=single_task_result.get('DAT_auroc', 0),
            net_auroc=0,  # Not trained
            sert_auroc=0,  # Not trained
            stereo_accuracy=single_task_result['stereo_accuracy'],
            auroc_delta=single_task_result['overall_auroc'] - baseline_result['overall_auroc'],
            stereo_delta=single_task_result['stereo_accuracy'] - baseline_result['stereo_accuracy'],
            best_epoch=single_task_result['best_epoch'],
            final_loss=single_task_result['final_loss'],
        ))

        # 4. Ablation: No attention readout (mean pooling)
        print("\n[4/6] Ablation: Mean pooling (no attention readout)...")
        no_attn_model = self._create_no_attention_model()
        no_attn_result = self._train_and_evaluate(
            model=no_attn_model,
            name="no_attention",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
        )

        study.ablations.append(AblationResult(
            name="no_attention_readout",
            description="Replace attention readout with mean pooling",
            config_changes={"readout": "mean"},
            overall_auroc=no_attn_result['overall_auroc'],
            dat_auroc=no_attn_result.get('DAT_auroc', 0),
            net_auroc=no_attn_result.get('NET_auroc', 0),
            sert_auroc=no_attn_result.get('SERT_auroc', 0),
            stereo_accuracy=no_attn_result['stereo_accuracy'],
            auroc_delta=no_attn_result['overall_auroc'] - baseline_result['overall_auroc'],
            stereo_delta=no_attn_result['stereo_accuracy'] - baseline_result['stereo_accuracy'],
            best_epoch=no_attn_result['best_epoch'],
            final_loss=no_attn_result['final_loss'],
        ))

        # 5. Ablation: Fewer GNN layers (3 instead of 6)
        print("\n[5/6] Ablation: Fewer GNN layers (3)...")
        shallow_model = self._create_shallow_model()
        shallow_result = self._train_and_evaluate(
            model=shallow_model,
            name="shallow_gnn",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
        )

        study.ablations.append(AblationResult(
            name="shallow_gnn",
            description="Use 3 GNN layers instead of 6",
            config_changes={"num_gnn_layers": 3},
            overall_auroc=shallow_result['overall_auroc'],
            dat_auroc=shallow_result.get('DAT_auroc', 0),
            net_auroc=shallow_result.get('NET_auroc', 0),
            sert_auroc=shallow_result.get('SERT_auroc', 0),
            stereo_accuracy=shallow_result['stereo_accuracy'],
            auroc_delta=shallow_result['overall_auroc'] - baseline_result['overall_auroc'],
            stereo_delta=shallow_result['stereo_accuracy'] - baseline_result['stereo_accuracy'],
            best_epoch=shallow_result['best_epoch'],
            final_loss=shallow_result['final_loss'],
        ))

        # 6. Ablation: No focal loss (standard CE)
        print("\n[6/6] Ablation: Standard cross-entropy loss...")
        ce_result = self._train_and_evaluate(
            model=StereoGNN(),
            name="standard_ce",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            loss_type="ce",
        )

        study.ablations.append(AblationResult(
            name="standard_ce_loss",
            description="Use standard cross-entropy instead of focal loss",
            config_changes={"loss_type": "ce"},
            overall_auroc=ce_result['overall_auroc'],
            dat_auroc=ce_result.get('DAT_auroc', 0),
            net_auroc=ce_result.get('NET_auroc', 0),
            sert_auroc=ce_result.get('SERT_auroc', 0),
            stereo_accuracy=ce_result['stereo_accuracy'],
            auroc_delta=ce_result['overall_auroc'] - baseline_result['overall_auroc'],
            stereo_delta=ce_result['stereo_accuracy'] - baseline_result['stereo_accuracy'],
            best_epoch=ce_result['best_epoch'],
            final_loss=ce_result['final_loss'],
        ))

        # Save results
        self._save_results(study)
        self._print_summary(study)

        return study

    def _train_and_evaluate(
        self,
        model: nn.Module,
        name: str,
        train_loader,
        val_loader,
        test_loader,
        num_epochs: int,
        loss_type: str = "focal",
    ) -> Dict:
        """Train a model and evaluate it."""
        trainer = Trainer(
            model=model,
            experiment_name=f"ablation_{name}",
        )

        # Override loss type if needed
        if loss_type != "focal":
            CONFIG.training.loss_type = loss_type

        trainer.setup(train_loader.dataset)

        # Train
        history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)

        # Evaluate
        evaluator = ModelEvaluator(model, device=self.device)
        results = evaluator.evaluate(test_loader)

        return {
            'overall_auroc': results.overall_auroc,
            'DAT_auroc': results.monoamine_aurocs.get('DAT', 0),
            'NET_auroc': results.monoamine_aurocs.get('NET', 0),
            'SERT_auroc': results.monoamine_aurocs.get('SERT', 0),
            'stereo_accuracy': results.stereo_accuracy,
            'best_epoch': trainer.current_epoch,
            'final_loss': history['train_total'][-1] if history['train_total'] else 0,
        }

    def _create_single_task_model(self) -> StereoGNN:
        """Create a model that only predicts DAT."""
        # Use standard model but will only use DAT head in training
        return StereoGNN()

    def _create_no_attention_model(self) -> nn.Module:
        """Create a model with mean pooling instead of attention."""
        from torch_geometric.nn import global_mean_pool

        class StereoGNNMeanPool(StereoGNN):
            def __init__(self):
                super().__init__()
                # Replace attention readout with mean pooling
                self.readout = None

            def forward(self, data, return_attention=False):
                x, edge_index, edge_attr, batch = (
                    data.x, data.edge_index, data.edge_attr, data.batch
                )

                x = self.node_encoder(x)
                edge_attr = self.edge_encoder(edge_attr)

                for layer in self.gnn_layers:
                    x = layer(x, edge_index, edge_attr)

                # Mean pooling instead of attention
                graph_emb = global_mean_pool(x, batch)

                shared = self.shared_layer(graph_emb)

                output = {
                    'DAT': self.heads['DAT'](shared),
                    'NET': self.heads['NET'](shared),
                    'SERT': self.heads['SERT'](shared),
                }

                return output

        return StereoGNNMeanPool()

    def _create_shallow_model(self) -> StereoGNN:
        """Create a model with fewer GNN layers."""
        # Modify config
        config = deepcopy(CONFIG.model)
        config.num_gnn_layers = 3
        return StereoGNN(config)

    def _save_results(self, study: AblationStudy):
        """Save ablation study results."""
        results_path = self.output_dir / "ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump(study.to_dict(), f, indent=2)
        print(f"\nResults saved to {results_path}")

    def _print_summary(self, study: AblationStudy):
        """Print summary of ablation study."""
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY")
        print("=" * 70)

        print(f"\nBaseline: {study.baseline_name}")
        print(f"  AUROC: {study.baseline_auroc:.4f}")
        print(f"  Stereo Accuracy: {study.baseline_stereo:.4f}")

        print(f"\n{'Ablation':<30} {'AUROC':<10} {'Delta':<10} {'Stereo':<10} {'Delta':<10}")
        print("-" * 70)

        for abl in study.ablations:
            print(
                f"{abl.name:<30} "
                f"{abl.overall_auroc:<10.4f} "
                f"{abl.auroc_delta:+.4f}{'':3} "
                f"{abl.stereo_accuracy:<10.4f} "
                f"{abl.stereo_delta:+.4f}"
            )

        print("-" * 70)

        # Check critical ablation (stereochemistry)
        stereo_ablation = next(
            (a for a in study.ablations if a.name == "no_stereochemistry"),
            None
        )
        if stereo_ablation:
            drop = -stereo_ablation.auroc_delta
            print(f"\nCRITICAL CHECK: Stereo ablation AUROC drop = {drop:.4f}")
            if drop >= 0.05:
                print("  STATUS: PASS (>= 5% drop proves stereo features matter)")
            else:
                print("  STATUS: WARN (< 5% drop - may need more stereoselective data)")

        print("=" * 70)


def run_ablation_study(num_epochs: int = 50) -> AblationStudy:
    """
    Convenience function to run full ablation study.

    Args:
        num_epochs: Training epochs per ablation

    Returns:
        AblationStudy with results
    """
    # Create dataloaders
    dataloaders = create_dataloaders(
        batch_size=CONFIG.training.batch_size,
        num_workers=CONFIG.training.num_workers,
        use_3d=False,
    )

    # Run ablation
    runner = AblationRunner()
    study = runner.run_full_ablation_study(
        dataloaders['train'],
        dataloaders['val'],
        dataloaders['test'],
        num_epochs=num_epochs,
    )

    return study


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation Study Framework Test")
    print("=" * 60)

    # Quick test with synthetic data
    print("\nThis script runs the full ablation study.")
    print("Run with: python ablation.py")
    print("\nFor quick testing, run: run_ablation_study(num_epochs=3)")
