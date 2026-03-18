"""
Comprehensive Evaluation Suite for StereoGNN
=============================================

Implements ALL success criteria metrics:
1. ROC-AUC (target: 0.85 overall, 0.95 monoamine-specific)
2. PR-AUC (target: >= 0.65)
3. Enrichment Factor @ 1%, 5% (target: EF@1% >= 10x)
4. Stereo sensitivity on known enantiomer pairs (target: >= 80%)
5. Calibration (ECE <= 0.10)
6. Virtual screening validation (rank known substrates > decoys)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from model import StereoGNN
from featurizer import MoleculeGraphFeaturizer
from config import CONFIG, EvaluationConfig


@dataclass
class EvaluationResults:
    """Container for all evaluation results."""

    # Primary metrics (MUST PASS)
    overall_auroc: float
    monoamine_aurocs: Dict[str, float]
    monoamine_praucs: Dict[str, float]

    # Stereo sensitivity
    stereo_accuracy: float
    stereo_pair_results: List[Dict]

    # Enrichment
    enrichment_factors: Dict[str, Dict[str, float]]

    # Calibration
    expected_calibration_error: float

    # Virtual screening
    virtual_screening_valid: bool
    vs_results: Dict

    # Success assessment
    passes_criteria: bool
    failed_criteria: List[str]

    def to_dict(self) -> Dict:
        return {
            'overall_auroc': self.overall_auroc,
            'monoamine_aurocs': self.monoamine_aurocs,
            'monoamine_praucs': self.monoamine_praucs,
            'stereo_accuracy': self.stereo_accuracy,
            'stereo_pair_results': self.stereo_pair_results,
            'enrichment_factors': self.enrichment_factors,
            'expected_calibration_error': self.expected_calibration_error,
            'virtual_screening_valid': self.virtual_screening_valid,
            'vs_results': self.vs_results,
            'passes_criteria': self.passes_criteria,
            'failed_criteria': self.failed_criteria,
        }


class MetricsCalculator:
    """Calculates evaluation metrics."""

    @staticmethod
    def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        if len(np.unique(y_true)) < 2:
            return 0.0
        try:
            return roc_auc_score(y_true, y_score)
        except Exception:
            return 0.0

    @staticmethod
    def prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve."""
        if len(np.unique(y_true)) < 2:
            return 0.0
        try:
            return average_precision_score(y_true, y_score)
        except Exception:
            return 0.0

    @staticmethod
    def enrichment_factor(
        y_true: np.ndarray,
        y_score: np.ndarray,
        fraction: float = 0.01,
    ) -> float:
        """
        Compute Enrichment Factor at given fraction.

        EF = (hits in top fraction) / (expected hits in random selection)
        """
        n = len(y_true)
        n_top = max(1, int(n * fraction))

        # Sort by score (descending)
        sorted_indices = np.argsort(y_score)[::-1]
        top_indices = sorted_indices[:n_top]

        # Count positives in top fraction
        hits_in_top = y_true[top_indices].sum()

        # Expected hits in random selection
        total_positives = y_true.sum()
        expected_hits = total_positives * fraction

        if expected_hits == 0:
            return 0.0

        return hits_in_top / expected_hits

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|

        Lower is better. Target: <= 0.10
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop_in_bin = in_bin.sum() / n

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence = y_prob[in_bin].mean()
                ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence)

        return ece


class StereoSensitivityEvaluator:
    """
    Evaluates model's ability to distinguish stereoisomers.

    Uses known enantiomer pairs with documented activity differences.
    """

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Known stereoselective pairs from config
        self.stereo_pairs = CONFIG.stereo.stereoselective_pairs

    def evaluate(self) -> Tuple[float, List[Dict]]:
        """
        Evaluate stereo sensitivity.

        Returns:
            Tuple of (accuracy, detailed_results)
            accuracy: Fraction of pairs correctly ranked
        """
        self.model.eval()
        results = []
        correct = 0

        for smiles1, smiles2, target, expected_ratio in self.stereo_pairs:
            # Featurize both stereoisomers
            data1 = self.featurizer.featurize(smiles1)
            data2 = self.featurizer.featurize(smiles2)

            if data1 is None or data2 is None:
                continue

            # Get predictions
            batch1 = Batch.from_data_list([data1]).to(self.device)
            batch2 = Batch.from_data_list([data2]).to(self.device)

            with torch.no_grad():
                pred1 = self.model(batch1)
                pred2 = self.model(batch2)

            # Get substrate probability for the relevant target
            prob1 = F.softmax(pred1[target], dim=-1)[0, 2].item()  # Substrate class
            prob2 = F.softmax(pred2[target], dim=-1)[0, 2].item()

            # Check if ranking is correct
            # expected_ratio > 1 means smiles1 should be more potent
            predicted_ratio = prob1 / (prob2 + 1e-8)
            is_correct = (expected_ratio > 1 and prob1 > prob2) or \
                        (expected_ratio < 1 and prob2 > prob1) or \
                        (expected_ratio == 1 and abs(prob1 - prob2) < 0.1)

            if is_correct:
                correct += 1

            results.append({
                'smiles1': smiles1,
                'smiles2': smiles2,
                'target': target,
                'expected_ratio': expected_ratio,
                'prob1': prob1,
                'prob2': prob2,
                'predicted_ratio': predicted_ratio,
                'correct': is_correct,
            })

        accuracy = correct / len(results) if results else 0.0
        return accuracy, results


class VirtualScreeningValidator:
    """
    Validates model for virtual screening use case.

    Known substrates should rank higher than:
    1. Known blockers
    2. Random decoys
    """

    def __init__(self, model: StereoGNN, device: torch.device):
        self.model = model
        self.device = device
        self.featurizer = MoleculeGraphFeaturizer(use_3d=False)

        # Known substrates and blockers from config
        self.substrates = CONFIG.evaluation.known_stimulant_substrates
        self.blockers = CONFIG.evaluation.known_blockers

    def evaluate(self) -> Tuple[bool, Dict]:
        """
        Run virtual screening validation.

        Returns:
            Tuple of (passes, detailed_results)
        """
        self.model.eval()
        results = {'substrates': [], 'blockers': []}

        # Score all substrates
        for smiles in self.substrates:
            data = self.featurizer.featurize(smiles)
            if data is None:
                continue

            batch = Batch.from_data_list([data]).to(self.device)

            with torch.no_grad():
                pred = self.model(batch)

            # Average substrate probability across all targets
            avg_prob = np.mean([
                F.softmax(pred[t], dim=-1)[0, 2].item()
                for t in ['DAT', 'NET', 'SERT']
            ])

            results['substrates'].append({
                'smiles': smiles,
                'avg_substrate_prob': avg_prob,
            })

        # Score all blockers
        for smiles in self.blockers:
            data = self.featurizer.featurize(smiles)
            if data is None:
                continue

            batch = Batch.from_data_list([data]).to(self.device)

            with torch.no_grad():
                pred = self.model(batch)

            avg_prob = np.mean([
                F.softmax(pred[t], dim=-1)[0, 2].item()
                for t in ['DAT', 'NET', 'SERT']
            ])

            results['blockers'].append({
                'smiles': smiles,
                'avg_substrate_prob': avg_prob,
            })

        # Check: substrates should have higher scores than blockers
        if not results['substrates'] or not results['blockers']:
            return False, results

        min_substrate = min(r['avg_substrate_prob'] for r in results['substrates'])
        max_blocker = max(r['avg_substrate_prob'] for r in results['blockers'])

        # Ideally, min substrate > max blocker
        # But we allow some overlap, requiring at least 80% separation
        substrate_probs = [r['avg_substrate_prob'] for r in results['substrates']]
        blocker_probs = [r['avg_substrate_prob'] for r in results['blockers']]

        # Calculate how many substrates rank above all blockers
        threshold = np.percentile(blocker_probs, 90)  # Top 10% of blockers
        substrates_above = sum(p > threshold for p in substrate_probs)
        separation_rate = substrates_above / len(substrate_probs)

        results['separation_rate'] = separation_rate
        results['min_substrate_prob'] = min_substrate
        results['max_blocker_prob'] = max_blocker

        passes = separation_rate >= 0.8  # 80% of substrates above 90th percentile blocker

        return passes, results


class ModelEvaluator:
    """
    Complete model evaluation pipeline.
    """

    def __init__(
        self,
        model: StereoGNN,
        config: EvaluationConfig = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config or CONFIG.evaluation
        self.device = device or torch.device(CONFIG.training.device)
        self.model = self.model.to(self.device)

        self.metrics = MetricsCalculator()
        self.stereo_eval = StereoSensitivityEvaluator(model, self.device)
        self.vs_eval = VirtualScreeningValidator(model, self.device)

    @torch.no_grad()
    def evaluate(self, dataloader) -> EvaluationResults:
        """
        Run complete evaluation.

        Args:
            dataloader: Test dataloader

        Returns:
            EvaluationResults with all metrics
        """
        self.model.eval()

        # Collect predictions
        all_predictions = {task: [] for task in ['DAT', 'NET', 'SERT']}
        all_targets = {task: [] for task in ['DAT', 'NET', 'SERT']}

        for batch in dataloader:
            batch = batch.to(self.device)
            predictions = self.model(batch)

            for task in all_predictions:
                probs = F.softmax(predictions[task], dim=-1)
                all_predictions[task].append(probs.cpu())

                target_attr = f'y_{task.lower()}'
                if hasattr(batch, target_attr):
                    all_targets[task].append(getattr(batch, target_attr).cpu())

        # Stack predictions
        for task in all_predictions:
            all_predictions[task] = torch.cat(all_predictions[task], dim=0).numpy()
            if all_targets[task]:
                all_targets[task] = torch.cat(all_targets[task], dim=0).numpy()
            else:
                all_targets[task] = np.array([])

        # Compute metrics per task
        monoamine_aurocs = {}
        monoamine_praucs = {}
        enrichment_factors = {}

        for task in ['DAT', 'NET', 'SERT']:
            preds = all_predictions[task]
            targs = all_targets[task].flatten()

            # Filter valid labels
            valid_mask = targs >= 0
            if valid_mask.sum() < 10:
                continue

            preds = preds[valid_mask]
            targs = targs[valid_mask]

            # Binary: substrate (2) vs not
            targs_binary = (targs == 2).astype(int)
            preds_substrate = preds[:, 2]

            # AUROC and PRAUC
            monoamine_aurocs[task] = self.metrics.auroc(targs_binary, preds_substrate)
            monoamine_praucs[task] = self.metrics.prauc(targs_binary, preds_substrate)

            # Enrichment factors
            enrichment_factors[task] = {}
            for frac in self.config.enrichment_factors:
                ef = self.metrics.enrichment_factor(targs_binary, preds_substrate, frac)
                enrichment_factors[task][f'EF@{int(frac*100)}%'] = ef

        # Overall AUROC
        overall_auroc = np.mean(list(monoamine_aurocs.values())) if monoamine_aurocs else 0.0

        # Calibration (on pooled predictions)
        all_binary_preds = []
        all_binary_targs = []
        for task in ['DAT', 'NET', 'SERT']:
            preds = all_predictions[task]
            targs = all_targets[task].flatten()
            valid_mask = targs >= 0
            if valid_mask.sum() > 0:
                all_binary_preds.extend(preds[valid_mask][:, 2])
                all_binary_targs.extend((targs[valid_mask] == 2).astype(int))

        if all_binary_preds:
            ece = self.metrics.expected_calibration_error(
                np.array(all_binary_targs),
                np.array(all_binary_preds),
            )
        else:
            ece = 1.0

        # Stereo sensitivity
        stereo_accuracy, stereo_results = self.stereo_eval.evaluate()

        # Virtual screening validation
        vs_passes, vs_results = self.vs_eval.evaluate()

        # Check success criteria
        failed_criteria = []

        if overall_auroc < self.config.min_overall_auroc:
            failed_criteria.append(
                f"Overall AUROC {overall_auroc:.4f} < {self.config.min_overall_auroc}"
            )

        for task, auroc in monoamine_aurocs.items():
            if auroc < self.config.min_monoamine_auroc:
                failed_criteria.append(
                    f"{task} AUROC {auroc:.4f} < {self.config.min_monoamine_auroc}"
                )

        for task, prauc in monoamine_praucs.items():
            if prauc < self.config.min_prauc:
                failed_criteria.append(
                    f"{task} PR-AUC {prauc:.4f} < {self.config.min_prauc}"
                )

        if stereo_accuracy < self.config.min_stereo_accuracy:
            failed_criteria.append(
                f"Stereo accuracy {stereo_accuracy:.4f} < {self.config.min_stereo_accuracy}"
            )

        if ece > self.config.max_ece:
            failed_criteria.append(
                f"ECE {ece:.4f} > {self.config.max_ece}"
            )

        # Check enrichment factor
        for task, efs in enrichment_factors.items():
            if efs.get('EF@1%', 0) < self.config.min_ef_1pct:
                failed_criteria.append(
                    f"{task} EF@1% {efs.get('EF@1%', 0):.2f} < {self.config.min_ef_1pct}"
                )

        if not vs_passes:
            failed_criteria.append("Virtual screening validation failed")

        passes = len(failed_criteria) == 0

        return EvaluationResults(
            overall_auroc=overall_auroc,
            monoamine_aurocs=monoamine_aurocs,
            monoamine_praucs=monoamine_praucs,
            stereo_accuracy=stereo_accuracy,
            stereo_pair_results=stereo_results,
            enrichment_factors=enrichment_factors,
            expected_calibration_error=ece,
            virtual_screening_valid=vs_passes,
            vs_results=vs_results,
            passes_criteria=passes,
            failed_criteria=failed_criteria,
        )

    def print_results(self, results: EvaluationResults):
        """Pretty print evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        print(f"\n{'METRIC':<30} {'VALUE':<15} {'TARGET':<15} {'STATUS':<10}")
        print("-" * 70)

        # Overall AUROC
        status = "PASS" if results.overall_auroc >= self.config.min_overall_auroc else "FAIL"
        print(f"{'Overall AUROC':<30} {results.overall_auroc:<15.4f} {'>= ' + str(self.config.min_overall_auroc):<15} {status:<10}")

        # Per-task metrics
        for task in ['DAT', 'NET', 'SERT']:
            if task in results.monoamine_aurocs:
                auroc = results.monoamine_aurocs[task]
                status = "PASS" if auroc >= self.config.min_monoamine_auroc else "FAIL"
                print(f"{task + ' AUROC':<30} {auroc:<15.4f} {'>= ' + str(self.config.min_monoamine_auroc):<15} {status:<10}")

            if task in results.monoamine_praucs:
                prauc = results.monoamine_praucs[task]
                status = "PASS" if prauc >= self.config.min_prauc else "FAIL"
                print(f"{task + ' PR-AUC':<30} {prauc:<15.4f} {'>= ' + str(self.config.min_prauc):<15} {status:<10}")

        # Stereo sensitivity
        status = "PASS" if results.stereo_accuracy >= self.config.min_stereo_accuracy else "FAIL"
        print(f"{'Stereo Sensitivity':<30} {results.stereo_accuracy:<15.4f} {'>= ' + str(self.config.min_stereo_accuracy):<15} {status:<10}")

        # Calibration
        status = "PASS" if results.expected_calibration_error <= self.config.max_ece else "FAIL"
        print(f"{'Calibration (ECE)':<30} {results.expected_calibration_error:<15.4f} {'<= ' + str(self.config.max_ece):<15} {status:<10}")

        # Virtual screening
        status = "PASS" if results.virtual_screening_valid else "FAIL"
        print(f"{'Virtual Screening':<30} {str(results.virtual_screening_valid):<15} {'True':<15} {status:<10}")

        # Enrichment factors
        print("\n" + "-" * 70)
        print("ENRICHMENT FACTORS")
        print("-" * 70)
        for task, efs in results.enrichment_factors.items():
            for ef_name, ef_val in efs.items():
                print(f"  {task} {ef_name}: {ef_val:.2f}x")

        # Final verdict
        print("\n" + "=" * 70)
        if results.passes_criteria:
            print("OVERALL: ALL CRITERIA PASSED")
        else:
            print("OVERALL: FAILED")
            print("\nFailed criteria:")
            for criterion in results.failed_criteria:
                print(f"  - {criterion}")
        print("=" * 70)

    def save_results(self, results: EvaluationResults, path: Path):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)


def evaluate_model(
    model_path: Path,
    test_loader,
    device: Optional[torch.device] = None,
) -> EvaluationResults:
    """
    Load a trained model and evaluate it.

    Args:
        model_path: Path to model checkpoint
        test_loader: Test dataloader
        device: Device to use

    Returns:
        EvaluationResults
    """
    device = device or torch.device(CONFIG.training.device)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = StereoGNN()
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    evaluator = ModelEvaluator(model, device=device)
    results = evaluator.evaluate(test_loader)
    evaluator.print_results(results)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Suite Test")
    print("=" * 60)

    # Test metrics calculator
    mc = MetricsCalculator()

    # Synthetic test data
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])

    print(f"\nAUROC: {mc.auroc(y_true, y_score):.4f}")
    print(f"PR-AUC: {mc.prauc(y_true, y_score):.4f}")
    print(f"EF@10%: {mc.enrichment_factor(y_true, y_score, 0.10):.2f}x")
    print(f"ECE: {mc.expected_calibration_error(y_true, y_score):.4f}")

    # Test stereo evaluator with dummy model
    print("\nTesting stereo sensitivity evaluator...")
    model = StereoGNN()
    device = torch.device('cpu')
    stereo_eval = StereoSensitivityEvaluator(model, device)
    accuracy, results = stereo_eval.evaluate()
    print(f"Stereo accuracy (untrained model): {accuracy:.4f}")
    print(f"Number of pairs evaluated: {len(results)}")
