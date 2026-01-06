"""
Visualization Module for StereoGNN
==================================

Generates:
1. ROC curves and PR curves per target
2. Confusion matrices
3. Embedding visualizations (t-SNE/UMAP)
4. Attention heatmaps on molecules
5. Ablation study plots
6. Stereo sensitivity analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'DAT': '#E74C3C',   # Red
    'NET': '#3498DB',   # Blue
    'SERT': '#2ECC71',  # Green
    'substrate': '#9B59B6',  # Purple
    'blocker': '#F39C12',    # Orange
    'inactive': '#95A5A6',   # Gray
}


class MetricsVisualizer:
    """Generate publication-quality metrics plots."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(
        self,
        predictions: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        title: str = "ROC Curves by Transporter",
        filename: str = "roc_curves.png",
    ):
        """
        Plot ROC curves for each transporter.

        Args:
            predictions: Dict mapping target -> predicted probabilities [N, 3]
            labels: Dict mapping target -> true labels [N]
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, target in enumerate(['DAT', 'NET', 'SERT']):
            ax = axes[idx]

            if target not in predictions:
                continue

            preds = predictions[target]
            lbls = labels[target]

            # One-vs-rest ROC for each class
            for class_idx, class_name in enumerate(['inactive', 'blocker', 'substrate']):
                binary_labels = (lbls == class_idx).astype(int)
                class_probs = preds[:, class_idx]

                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                roc_auc = auc(fpr, tpr)

                ax.plot(
                    fpr, tpr,
                    color=COLORS[class_name],
                    label=f'{class_name} (AUC = {roc_auc:.3f})',
                    linewidth=2,
                )

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{target}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pr_curves(
        self,
        predictions: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        filename: str = "pr_curves.png",
    ):
        """
        Plot Precision-Recall curves (focusing on substrate class).
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, target in enumerate(['DAT', 'NET', 'SERT']):
            ax = axes[idx]

            if target not in predictions:
                continue

            preds = predictions[target]
            lbls = labels[target]

            # Substrate class (class 2)
            binary_labels = (lbls == 2).astype(int)
            substrate_probs = preds[:, 2]

            precision, recall, _ = precision_recall_curve(binary_labels, substrate_probs)
            pr_auc = auc(recall, precision)

            ax.plot(
                recall, precision,
                color=COLORS[target],
                label=f'PR-AUC = {pr_auc:.3f}',
                linewidth=2,
            )

            # Baseline
            baseline = binary_labels.mean()
            ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5,
                      label=f'Baseline = {baseline:.3f}')

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'{target} - Substrate Detection', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)

        plt.suptitle('Precision-Recall Curves for Substrate Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(
        self,
        predictions: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        filename: str = "confusion_matrices.png",
    ):
        """Plot confusion matrices for each transporter."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        class_names = ['Inactive', 'Blocker', 'Substrate']

        for idx, target in enumerate(['DAT', 'NET', 'SERT']):
            ax = axes[idx]

            if target not in predictions:
                continue

            preds = predictions[target].argmax(axis=1)
            lbls = labels[target]

            cm = confusion_matrix(lbls, preds)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar=False,
            )

            # Add counts
            for i in range(3):
                for j in range(3):
                    ax.text(
                        j + 0.5, i + 0.7,
                        f'(n={cm[i, j]})',
                        ha='center', va='center',
                        fontsize=8, color='gray'
                    )

            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)
            ax.set_title(f'{target}', fontsize=14, fontweight='bold')

        plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_metrics_table(
        self,
        predictions: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        filename: str = "metrics_table.csv",
    ) -> pd.DataFrame:
        """Create comprehensive metrics table."""
        results = []

        for target in ['DAT', 'NET', 'SERT']:
            if target not in predictions:
                continue

            preds = predictions[target]
            lbls = labels[target]
            pred_classes = preds.argmax(axis=1)

            # Per-class metrics
            for class_idx, class_name in enumerate(['Inactive', 'Blocker', 'Substrate']):
                binary_labels = (lbls == class_idx).astype(int)
                class_probs = preds[:, class_idx]

                # ROC-AUC
                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                roc_auc = auc(fpr, tpr)

                # PR-AUC
                precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
                pr_auc_score = auc(recall, precision)

                # Accuracy for this class
                class_acc = ((pred_classes == class_idx) == (lbls == class_idx)).mean()

                results.append({
                    'Target': target,
                    'Class': class_name,
                    'ROC-AUC': roc_auc,
                    'PR-AUC': pr_auc_score,
                    'Accuracy': class_acc,
                    'Support': int(binary_labels.sum()),
                })

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / filename, index=False)

        return df


class EmbeddingVisualizer:
    """Visualize learned molecule embeddings."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_embeddings_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str] = None,
        title: str = "Molecule Embeddings (t-SNE)",
        filename: str = "embeddings_tsne.png",
        perplexity: int = 30,
    ):
        """
        Plot embeddings using t-SNE.

        Args:
            embeddings: [N, D] array of embeddings
            labels: [N] array of labels
            label_names: Names for each label value
        """
        label_names = label_names or ['Inactive', 'Blocker', 'Substrate']

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))

        for label_idx, label_name in enumerate(label_names):
            mask = labels == label_idx
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=label_name,
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_embeddings_umap(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        label_names: List[str] = None,
        title: str = "Molecule Embeddings (UMAP)",
        filename: str = "embeddings_umap.png",
    ):
        """Plot embeddings using UMAP."""
        if not UMAP_AVAILABLE:
            print("UMAP not available, skipping")
            return

        label_names = label_names or ['Inactive', 'Blocker', 'Substrate']

        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(10, 10))

        for label_idx, label_name in enumerate(label_names):
            mask = labels == label_idx
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=label_name,
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stereo_embedding_comparison(
        self,
        d_embeddings: np.ndarray,
        l_embeddings: np.ndarray,
        pair_names: List[str],
        filename: str = "stereo_embeddings.png",
    ):
        """
        Visualize how enantiomers are separated in embedding space.

        Args:
            d_embeddings: Embeddings for d-isomers
            l_embeddings: Embeddings for l-isomers
            pair_names: Names of the compound pairs
        """
        # Combine for t-SNE
        all_embeddings = np.vstack([d_embeddings, l_embeddings])
        all_labels = ['d-'] * len(d_embeddings) + ['l-'] * len(l_embeddings)

        tsne = TSNE(n_components=2, perplexity=min(5, len(all_embeddings) - 1), random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        fig, ax = plt.subplots(figsize=(10, 10))

        n = len(d_embeddings)

        # Plot d-isomers
        ax.scatter(
            embeddings_2d[:n, 0],
            embeddings_2d[:n, 1],
            c='blue', label='d-isomer', s=100, marker='o'
        )

        # Plot l-isomers
        ax.scatter(
            embeddings_2d[n:, 0],
            embeddings_2d[n:, 1],
            c='red', label='l-isomer', s=100, marker='^'
        )

        # Draw lines between pairs
        for i in range(n):
            ax.plot(
                [embeddings_2d[i, 0], embeddings_2d[n+i, 0]],
                [embeddings_2d[i, 1], embeddings_2d[n+i, 1]],
                'gray', linestyle='--', alpha=0.5
            )

            # Label the pair
            mid_x = (embeddings_2d[i, 0] + embeddings_2d[n+i, 0]) / 2
            mid_y = (embeddings_2d[i, 1] + embeddings_2d[n+i, 1]) / 2
            ax.annotate(pair_names[i], (mid_x, mid_y), fontsize=8)

        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('Stereoisomer Separation in Embedding Space', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()


class AttentionVisualizer:
    """Visualize attention weights on molecules."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_atom_attention(
        self,
        smiles: str,
        attention_weights: np.ndarray,
        title: str = None,
        filename: str = "atom_attention.png",
    ):
        """
        Plot attention weights on molecule structure.

        Args:
            smiles: SMILES string
            attention_weights: [N_atoms] attention weights
        """
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        # Normalize attention
        attention_weights = np.array(attention_weights)
        attention_normalized = (attention_weights - attention_weights.min()) / \
                              (attention_weights.max() - attention_weights.min() + 1e-8)

        # Create atom colors based on attention
        atom_colors = {}
        for i, attn in enumerate(attention_normalized):
            # Red = high attention, blue = low
            atom_colors[i] = (attn, 0, 1 - attn)

        # Draw with highlights
        fig = Draw.MolToMPL(mol, size=(400, 400), highlightAtoms=list(range(mol.GetNumAtoms())))

        title = title or f"Attention: {smiles[:30]}..."
        plt.title(title, fontsize=12)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()


class AblationVisualizer:
    """Visualize ablation study results."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_ablation_comparison(
        self,
        full_metrics: Dict[str, float],
        ablation_metrics: Dict[str, float],
        filename: str = "ablation_comparison.png",
    ):
        """
        Plot comparison between full model and ablation model.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC-AUC comparison
        ax1 = axes[0]
        targets = ['DAT', 'NET', 'SERT']
        x = np.arange(len(targets))
        width = 0.35

        full_aucs = [full_metrics.get(f'{t}_auc', 0) for t in targets]
        abl_aucs = [ablation_metrics.get(f'{t}_auc', 0) for t in targets]

        ax1.bar(x - width/2, full_aucs, width, label='Full Model', color=COLORS['substrate'])
        ax1.bar(x + width/2, abl_aucs, width, label='No Stereo Features', color=COLORS['inactive'])

        ax1.set_ylabel('ROC-AUC', fontsize=12)
        ax1.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets)
        ax1.legend()
        ax1.set_ylim([0.5, 1.0])

        # Calculate drops
        drops = [full_aucs[i] - abl_aucs[i] for i in range(len(targets))]
        for i, drop in enumerate(drops):
            ax1.annotate(
                f'Δ = {drop:.3f}',
                xy=(x[i], max(full_aucs[i], abl_aucs[i]) + 0.02),
                ha='center', fontsize=10,
            )

        # PR-AUC comparison
        ax2 = axes[1]

        full_pr = [full_metrics.get(f'{t}_pr_auc', 0) for t in targets]
        abl_pr = [ablation_metrics.get(f'{t}_pr_auc', 0) for t in targets]

        ax2.bar(x - width/2, full_pr, width, label='Full Model', color=COLORS['substrate'])
        ax2.bar(x + width/2, abl_pr, width, label='No Stereo Features', color=COLORS['inactive'])

        ax2.set_ylabel('PR-AUC', fontsize=12)
        ax2.set_title('PR-AUC Comparison (Substrate)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(targets)
        ax2.legend()
        ax2.set_ylim([0, 1.0])

        plt.suptitle('Ablation Study: Impact of Stereochemistry Features',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def create_ablation_table(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str = "ablation_table.csv",
    ) -> pd.DataFrame:
        """Create ablation study results table."""
        rows = []

        for model_name, metrics in results.items():
            for target in ['DAT', 'NET', 'SERT']:
                rows.append({
                    'Model': model_name,
                    'Target': target,
                    'ROC-AUC': metrics.get(f'{target}_auc', 0),
                    'PR-AUC': metrics.get(f'{target}_pr_auc', 0),
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / filename, index=False)

        return df


def generate_all_visualizations(
    model,
    test_loader,
    output_dir: Path,
    device: torch.device = None,
):
    """
    Generate all visualizations for a trained model.
    """
    output_dir = Path(output_dir)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    # Collect predictions and embeddings
    all_preds = {t: [] for t in ['DAT', 'NET', 'SERT']}
    all_labels = {t: [] for t in ['DAT', 'NET', 'SERT']}
    all_embeddings = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch, return_attention=True)

            for target in ['DAT', 'NET', 'SERT']:
                if hasattr(batch, f'{target}_label'):
                    labels = getattr(batch, f'{target}_label')
                    mask = labels >= 0

                    if mask.sum() > 0:
                        probs = F.softmax(outputs[target][mask], dim=-1)
                        all_preds[target].append(probs.cpu().numpy())
                        all_labels[target].append(labels[mask].cpu().numpy())

            if 'graph_embedding' in outputs:
                all_embeddings.append(outputs['graph_embedding'].cpu().numpy())

    # Concatenate
    for target in ['DAT', 'NET', 'SERT']:
        if all_preds[target]:
            all_preds[target] = np.vstack(all_preds[target])
            all_labels[target] = np.concatenate(all_labels[target])

    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)

    # Generate visualizations
    metrics_viz = MetricsVisualizer(output_dir / 'metrics')
    metrics_viz.plot_roc_curves(all_preds, all_labels)
    metrics_viz.plot_pr_curves(all_preds, all_labels)
    metrics_viz.plot_confusion_matrices(all_preds, all_labels)
    metrics_table = metrics_viz.create_metrics_table(all_preds, all_labels)

    if len(all_embeddings) > 0 and len(all_labels.get('DAT', [])) > 0:
        embed_viz = EmbeddingVisualizer(output_dir / 'embeddings')
        embed_viz.plot_embeddings_tsne(all_embeddings, all_labels['DAT'])

    print(f"Visualizations saved to {output_dir}")

    return metrics_table


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)

    output_dir = Path("./outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic predictions
    n = 500
    predictions = {
        'DAT': np.random.rand(n, 3),
        'NET': np.random.rand(n, 3),
        'SERT': np.random.rand(n, 3),
    }

    # Normalize to probabilities
    for t in predictions:
        predictions[t] = predictions[t] / predictions[t].sum(axis=1, keepdims=True)

    labels = {
        'DAT': np.random.randint(0, 3, n),
        'NET': np.random.randint(0, 3, n),
        'SERT': np.random.randint(0, 3, n),
    }

    # Generate plots
    viz = MetricsVisualizer(output_dir)
    viz.plot_roc_curves(predictions, labels)
    viz.plot_pr_curves(predictions, labels)
    viz.plot_confusion_matrices(predictions, labels)
    df = viz.create_metrics_table(predictions, labels)

    print("Demo visualizations generated")
    print(df)
