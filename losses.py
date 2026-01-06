"""
Loss Functions for StereoGNN
============================

Custom loss functions for handling:
1. Class imbalance (Focal Loss)
2. Multi-task learning
3. Uncertainty-aware training
4. Label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 reduces the relative loss for well-classified examples,
    focusing training on hard negatives.

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -1,
    ):
        """
        Args:
            alpha: Class weights [num_classes]. If None, uniform weights.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Label to ignore (e.g., missing labels)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            inputs: Logits [N, C]
            targets: Labels [N]

        Returns:
            Loss value
        """
        # Handle ignore index
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device)

        # Softmax probabilities
        probs = F.softmax(inputs, dim=-1)

        # Get probability of correct class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross entropy loss.

    Prevents overconfident predictions by softening the target distribution.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 3,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Handle ignore index
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device)

        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        return (-true_dist * log_probs).sum(dim=-1).mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for DAT, NET, SERT prediction.

    Combines individual task losses with learned or fixed weights.
    Supports uncertainty-weighted loss (Kendall et al., 2018).
    """

    def __init__(
        self,
        loss_fn: str = 'focal',
        task_weights: Optional[Dict[str, float]] = None,
        learn_weights: bool = True,
        focal_gamma: float = 2.0,
        class_weights: Optional[Dict[str, Tensor]] = None,
        ignore_index: int = -1,
    ):
        """
        Args:
            loss_fn: 'focal', 'ce', or 'label_smooth'
            task_weights: Fixed weights per task
            learn_weights: Whether to learn task weights
            focal_gamma: Gamma for focal loss
            class_weights: Per-class weights for each task
            ignore_index: Label to ignore
        """
        super().__init__()

        self.task_names = ['DAT', 'NET', 'SERT']
        self.ignore_index = ignore_index

        # Initialize task-specific loss functions
        self.losses = nn.ModuleDict()
        for task in self.task_names:
            cw = class_weights.get(task) if class_weights else None

            if loss_fn == 'focal':
                self.losses[task] = FocalLoss(
                    alpha=cw,
                    gamma=focal_gamma,
                    ignore_index=ignore_index,
                )
            elif loss_fn == 'label_smooth':
                self.losses[task] = LabelSmoothingLoss(
                    smoothing=0.1,
                    ignore_index=ignore_index,
                )
            else:
                self.losses[task] = nn.CrossEntropyLoss(
                    weight=cw,
                    ignore_index=ignore_index,
                )

        # Task weights
        if learn_weights:
            # Learnable log-variance for uncertainty weighting
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in self.task_names
            })
        else:
            self.log_vars = None

        self.fixed_weights = task_weights or {t: 1.0 for t in self.task_names}

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dict of logits per task
            targets: Dict of labels per task

        Returns:
            Dict with 'total' loss and individual task losses
        """
        task_losses = {}
        total_loss = 0.0

        for task in self.task_names:
            pred = predictions[task]
            target = targets[task].squeeze()

            # Skip if all labels are ignore_index
            if (target == self.ignore_index).all():
                task_losses[task] = torch.tensor(0.0, device=pred.device)
                continue

            loss = self.losses[task](pred, target)
            task_losses[task] = loss

            if self.log_vars is not None:
                # Uncertainty weighting: L = 1/(2*sigma^2) * loss + log(sigma)
                log_var = self.log_vars[task]
                precision = torch.exp(-log_var)
                total_loss = total_loss + precision * loss + log_var
            else:
                total_loss = total_loss + self.fixed_weights[task] * loss

        task_losses['total'] = total_loss
        return task_losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better stereo representations.

    Pulls together embeddings of similar compounds (same target activity)
    and pushes apart embeddings of different compounds.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Args:
            embeddings: [N, D] normalized embeddings
            labels: [N] class labels

        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.t()).float()

        # Remove diagonal
        pos_mask.fill_diagonal_(0)

        # Negative mask
        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)

        # InfoNCE-style loss
        exp_sim = torch.exp(sim_matrix)

        # Positive pair loss
        pos_loss = -torch.log(
            (exp_sim * pos_mask).sum(dim=1) /
            (exp_sim * neg_mask + exp_sim * pos_mask).sum(dim=1) + 1e-8
        )

        # Only compute loss for samples with at least one positive pair
        valid_mask = pos_mask.sum(dim=1) > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return pos_loss[valid_mask].mean()


class StereoContrastiveLoss(nn.Module):
    """
    Contrastive loss specifically for stereoisomer pairs.

    Key insight: Enantiomers should have DIFFERENT embeddings
    (they have different biological activity).

    This loss encourages the model to distinguish stereoisomers.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: Tensor,
        emb2: Tensor,
        same_activity: Tensor,
    ) -> Tensor:
        """
        Args:
            emb1: Embeddings of first stereoisomer [N, D]
            emb2: Embeddings of second stereoisomer [N, D]
            same_activity: 1 if same activity, 0 if different [N]

        Returns:
            Loss encouraging correct embedding distance
        """
        # Cosine distance
        cos_sim = F.cosine_similarity(emb1, emb2)
        distance = 1 - cos_sim

        # If same activity: minimize distance
        # If different activity: maximize distance (with margin)
        same_loss = same_activity * distance ** 2
        diff_loss = (1 - same_activity) * F.relu(self.margin - distance) ** 2

        return (same_loss + diff_loss).mean()


def compute_class_weights(
    labels: Tensor,
    num_classes: int = 3,
    ignore_index: int = -1,
) -> Tensor:
    """
    Compute inverse frequency class weights for imbalanced data.
    """
    # Filter out ignored labels
    valid_labels = labels[labels != ignore_index]

    if len(valid_labels) == 0:
        return torch.ones(num_classes)

    # Count each class
    counts = torch.bincount(valid_labels.long(), minlength=num_classes).float()

    # Inverse frequency weighting
    # Add small epsilon to avoid division by zero
    weights = len(valid_labels) / (num_classes * counts + 1e-6)

    # Normalize so mean weight is 1
    weights = weights / weights.mean()

    return weights


# =============================================================================
# KINETIC EXTENSION - Loss functions for mechanistic parameter prediction
# =============================================================================

class GaussianNLLLoss(nn.Module):
    """
    Heteroscedastic Gaussian negative log-likelihood loss.

    For regression with predicted uncertainty:
        Loss = 0.5 * (log_var + (target - mean)^2 / exp(log_var))

    This allows the model to learn:
    1. The prediction (mean)
    2. The data noise/uncertainty (variance)

    Benefits:
    - Automatically down-weights uncertain predictions
    - Learns which samples have inherently noisy labels
    - Provides calibrated uncertainty estimates
    """

    def __init__(
        self,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.reduction = reduction

    def forward(
        self,
        pred_mean: Tensor,
        pred_log_var: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Gaussian NLL loss.

        Args:
            pred_mean: Predicted means [N]
            pred_log_var: Predicted log variances [N]
            target: Target values [N]
            mask: Optional boolean mask for valid samples [N]

        Returns:
            Loss value
        """
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred_mean.device, requires_grad=True)
            pred_mean = pred_mean[mask]
            pred_log_var = pred_log_var[mask]
            target = target[mask]

        # Clamp log_var for numerical stability
        pred_log_var = torch.clamp(pred_log_var, self.min_log_var, self.max_log_var)

        # Gaussian NLL: 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        precision = torch.exp(-pred_log_var)
        loss = 0.5 * (pred_log_var + precision * (target - pred_mean) ** 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class KineticMultiTaskLoss(nn.Module):
    """
    Combined loss for classification + kinetic regression with uncertainty.

    Handles:
    1. Activity classification (substrate/blocker/inactive) - Focal Loss
    2. Interaction mode classification (4-class) - Cross Entropy
    3. Kinetic regression (pKi, pIC50, bias) - Gaussian NLL

    Uses homoscedastic uncertainty weighting to automatically balance task losses.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
    """

    def __init__(
        self,
        learn_weights: bool = True,
        focal_gamma: float = 2.0,
        class_weights: Optional[Dict[str, Tensor]] = None,
        ignore_index: int = -1,
    ):
        """
        Args:
            learn_weights: Whether to learn task weights via homoscedastic uncertainty
            focal_gamma: Gamma parameter for focal loss
            class_weights: Optional per-class weights for classification
            ignore_index: Label value to ignore
        """
        super().__init__()

        self.tasks = ['DAT', 'NET', 'SERT']
        self.ignore_index = ignore_index

        # Classification losses
        self.cls_loss = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)
        self.mode_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Regression loss (Gaussian NLL for heteroscedastic uncertainty)
        self.regression_loss = GaussianNLLLoss()

        # Learnable task weights (homoscedastic uncertainty)
        if learn_weights:
            self.log_vars = nn.ParameterDict({
                'cls': nn.Parameter(torch.zeros(1)),
                'pKi': nn.Parameter(torch.zeros(1)),
                'pIC50': nn.Parameter(torch.zeros(1)),
                'bias': nn.Parameter(torch.zeros(1)),
                'mode': nn.Parameter(torch.zeros(1)),
            })
        else:
            self.log_vars = None

        # Store class weights
        self.class_weights = class_weights

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute multi-task loss for kinetic model.

        Args:
            predictions: Model outputs dict containing:
                - '{task}': Activity classification logits [batch, 3]
                - '{task}_pKi_mean': Predicted pKi [batch]
                - '{task}_pKi_log_var': Predicted pKi uncertainty [batch]
                - '{task}_pIC50_mean': Predicted pIC50 [batch]
                - '{task}_pIC50_log_var': Predicted pIC50 uncertainty [batch]
                - '{task}_kinetic_bias_mean': Predicted bias [batch]
                - '{task}_kinetic_bias_log_var': Predicted bias uncertainty [batch]
                - '{task}_interaction_mode': Mode logits [batch, 4]

            targets: Target dict containing:
                - '{task}' or '{task}_class': Activity labels [batch]
                - '{task}_pKi': pKi values (NaN for missing) [batch]
                - '{task}_pIC50': pIC50 values (NaN for missing) [batch]
                - '{task}_kinetic_bias': Bias values (NaN for missing) [batch]
                - '{task}_mode': Mode labels (-1 for missing) [batch]

        Returns:
            Dict with 'total' loss and individual task/component losses
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for task in self.tasks:
            # === 1. Activity classification loss ===
            cls_pred = predictions.get(task)
            cls_target = targets.get(f'{task}_class', targets.get(task))

            if cls_pred is not None and cls_target is not None:
                # Handle tensor conversion if needed
                if not isinstance(cls_target, Tensor):
                    cls_target = torch.tensor(cls_target, device=cls_pred.device)
                cls_target = cls_target.squeeze()

                mask = cls_target != self.ignore_index
                if mask.any():
                    l_cls = self.cls_loss(cls_pred, cls_target)
                    losses[f'{task}_cls'] = l_cls

                    if self.log_vars is not None:
                        precision = torch.exp(-self.log_vars['cls'])
                        total = total + precision * l_cls
                    else:
                        total = total + l_cls

            # === 2. Kinetic regression losses ===

            # pKi
            pki_target = targets.get(f'{task}_pKi')
            if pki_target is not None and f'{task}_pKi_mean' in predictions:
                if not isinstance(pki_target, Tensor):
                    pki_target = torch.tensor(pki_target, device=predictions[f'{task}_pKi_mean'].device)

                mask = ~torch.isnan(pki_target)
                if mask.any():
                    l_pki = self.regression_loss(
                        predictions[f'{task}_pKi_mean'],
                        predictions[f'{task}_pKi_log_var'],
                        pki_target,
                        mask
                    )
                    losses[f'{task}_pKi'] = l_pki

                    if self.log_vars is not None:
                        precision = torch.exp(-self.log_vars['pKi'])
                        total = total + precision * l_pki
                    else:
                        total = total + l_pki

            # pIC50
            pic50_target = targets.get(f'{task}_pIC50')
            if pic50_target is not None and f'{task}_pIC50_mean' in predictions:
                if not isinstance(pic50_target, Tensor):
                    pic50_target = torch.tensor(pic50_target, device=predictions[f'{task}_pIC50_mean'].device)

                mask = ~torch.isnan(pic50_target)
                if mask.any():
                    l_pic50 = self.regression_loss(
                        predictions[f'{task}_pIC50_mean'],
                        predictions[f'{task}_pIC50_log_var'],
                        pic50_target,
                        mask
                    )
                    losses[f'{task}_pIC50'] = l_pic50

                    if self.log_vars is not None:
                        precision = torch.exp(-self.log_vars['pIC50'])
                        total = total + precision * l_pic50
                    else:
                        total = total + l_pic50

            # Kinetic bias
            bias_target = targets.get(f'{task}_kinetic_bias')
            if bias_target is not None and f'{task}_kinetic_bias_mean' in predictions:
                if not isinstance(bias_target, Tensor):
                    bias_target = torch.tensor(bias_target, device=predictions[f'{task}_kinetic_bias_mean'].device)

                mask = ~torch.isnan(bias_target)
                if mask.any():
                    l_bias = self.regression_loss(
                        predictions[f'{task}_kinetic_bias_mean'],
                        predictions[f'{task}_kinetic_bias_log_var'],
                        bias_target,
                        mask
                    )
                    losses[f'{task}_bias'] = l_bias

                    if self.log_vars is not None:
                        precision = torch.exp(-self.log_vars['bias'])
                        total = total + precision * l_bias
                    else:
                        total = total + l_bias

            # === 3. Interaction mode classification ===
            mode_target = targets.get(f'{task}_mode')
            if mode_target is not None and f'{task}_interaction_mode' in predictions:
                if not isinstance(mode_target, Tensor):
                    mode_target = torch.tensor(mode_target, device=predictions[f'{task}_interaction_mode'].device)
                mode_target = mode_target.squeeze().long()

                mask = mode_target != self.ignore_index
                if mask.any():
                    l_mode = self.mode_loss(
                        predictions[f'{task}_interaction_mode'][mask],
                        mode_target[mask]
                    )
                    losses[f'{task}_mode'] = l_mode

                    if self.log_vars is not None:
                        precision = torch.exp(-self.log_vars['mode'])
                        total = total + precision * l_mode
                    else:
                        total = total + l_mode

        # Add regularization from learned weights (prevents all weights → infinity)
        if self.log_vars is not None:
            for name, log_var in self.log_vars.items():
                total = total + 0.5 * log_var

        losses['total'] = total

        return losses

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (for logging/debugging)."""
        if self.log_vars is None:
            return {name: 1.0 for name in ['cls', 'pKi', 'pIC50', 'bias', 'mode']}

        return {
            name: torch.exp(-log_var).item()
            for name, log_var in self.log_vars.items()
        }


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for relative potency prediction.

    Useful when absolute Ki/IC50 values are uncertain but relative
    ordering is known (e.g., compound A is more potent than B).

    Loss = max(0, margin - (pred_more_potent - pred_less_potent))
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pred_more_potent: Tensor,
        pred_less_potent: Tensor,
    ) -> Tensor:
        """
        Args:
            pred_more_potent: Predictions for more potent compounds [N]
            pred_less_potent: Predictions for less potent compounds [N]

        Returns:
            Ranking loss
        """
        # We want pred_more_potent > pred_less_potent by at least margin
        diff = pred_more_potent - pred_less_potent
        loss = F.relu(self.margin - diff)
        return loss.mean()


class UncertaintyRegularizer(nn.Module):
    """
    Regularization to prevent uncertainty collapse or explosion.

    Encourages predicted uncertainties to be in a reasonable range
    and match the actual prediction errors.
    """

    def __init__(
        self,
        target_log_var: float = 0.0,
        weight: float = 0.01,
    ):
        super().__init__()
        self.target_log_var = target_log_var
        self.weight = weight

    def forward(self, pred_log_var: Tensor) -> Tensor:
        """
        Regularize predicted log variances toward target.

        Args:
            pred_log_var: Predicted log variances [N]

        Returns:
            Regularization loss
        """
        return self.weight * (pred_log_var - self.target_log_var).pow(2).mean()


class EnantiomerPairLoss(nn.Module):
    """
    Loss for enforcing correct relative activity between enantiomer pairs.

    For amphetamines: d-isomer (R) typically >> l-isomer (S) for DAT/NET activity.
    This loss enforces that the model learns this stereochemical preference.

    Given pairs (active_isomer, less_active_isomer):
    - Classification: active_isomer should have higher predicted activity class
    - Regression: active_isomer should have higher pKi/pIC50 prediction

    Usage:
        loss_fn = EnantiomerPairLoss(margin=0.5)
        loss = loss_fn(pred_active, pred_inactive, activity_ratio=10.0)
    """

    def __init__(
        self,
        margin: float = 0.5,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
    ):
        super().__init__()
        self.margin = margin
        self.cls_weight = classification_weight
        self.reg_weight = regression_weight
        self.ranking_loss = RankingLoss(margin=margin)

    def forward(
        self,
        pred_active: Dict[str, Tensor],
        pred_inactive: Dict[str, Tensor],
        target: str = 'DAT',
        activity_ratio: float = 5.0,
    ) -> Dict[str, Tensor]:
        """
        Compute enantiomer pair loss.

        Args:
            pred_active: Predictions for more active isomer (e.g., d-amphetamine)
                Keys: '{target}_logits', '{target}_pKi_mean', etc.
            pred_inactive: Predictions for less active isomer (e.g., l-amphetamine)
            target: Target transporter (DAT, NET, SERT)
            activity_ratio: Expected activity ratio (active/inactive)

        Returns:
            Dict with 'total', 'classification', 'regression' losses
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(pred_active.values())).device)

        # 1. Classification ranking: active isomer should have higher class probability
        logits_key = f'{target}_logits'
        if logits_key in pred_active and logits_key in pred_inactive:
            # For substrate prediction, class 2 = substrate (most active)
            # We want P(substrate | active_isomer) > P(substrate | inactive_isomer)
            prob_active = F.softmax(pred_active[logits_key], dim=-1)[:, 2]  # P(substrate)
            prob_inactive = F.softmax(pred_inactive[logits_key], dim=-1)[:, 2]

            cls_loss = self.ranking_loss(prob_active, prob_inactive)
            losses['classification'] = cls_loss
            total = total + self.cls_weight * cls_loss

        # 2. pKi ranking: active isomer should have higher binding affinity
        pki_key = f'{target}_pKi_mean'
        if pki_key in pred_active and pki_key in pred_inactive:
            # Higher pKi = stronger binding
            pki_active = pred_active[pki_key].squeeze()
            pki_inactive = pred_inactive[pki_key].squeeze()

            # Margin based on expected ratio: log10(ratio) ~ 0.7 for 5x difference
            expected_diff = torch.log10(torch.tensor(activity_ratio))
            pki_loss = F.relu(expected_diff - (pki_active - pki_inactive)).mean()
            losses['pKi_ranking'] = pki_loss
            total = total + self.reg_weight * pki_loss

        # 3. pIC50 ranking
        pic50_key = f'{target}_pIC50_mean'
        if pic50_key in pred_active and pic50_key in pred_inactive:
            pic50_active = pred_active[pic50_key].squeeze()
            pic50_inactive = pred_inactive[pic50_key].squeeze()

            expected_diff = torch.log10(torch.tensor(activity_ratio))
            pic50_loss = F.relu(expected_diff - (pic50_active - pic50_inactive)).mean()
            losses['pIC50_ranking'] = pic50_loss
            total = total + self.reg_weight * pic50_loss

        losses['total'] = total
        return losses


# =============================================================================
# KNOWN ENANTIOMER PAIRS FOR TRAINING AUGMENTATION
# =============================================================================

ENANTIOMER_PAIRS = [
    # (active_smiles, inactive_smiles, target, activity_ratio)
    # Amphetamines - d-isomer more active at DAT/NET
    ("C[C@H](N)Cc1ccccc1", "C[C@@H](N)Cc1ccccc1", "DAT", 10.0),  # d vs l amphetamine
    ("C[C@H](N)Cc1ccccc1", "C[C@@H](N)Cc1ccccc1", "NET", 5.0),
    ("C[C@H](NC)Cc1ccccc1", "C[C@@H](NC)Cc1ccccc1", "DAT", 8.0),  # d vs l methamphetamine
    ("C[C@H](NC)Cc1ccccc1", "C[C@@H](NC)Cc1ccccc1", "NET", 4.0),
    # MDMA - S-isomer more active at SERT
    ("C[C@H](NC)Cc1ccc2OCOc2c1", "C[C@@H](NC)Cc1ccc2OCOc2c1", "SERT", 3.0),
    # Cathinone
    ("C[C@H](N)C(=O)c1ccccc1", "C[C@@H](N)C(=O)c1ccccc1", "DAT", 5.0),
    # Methylphenidate
    ("COC(=O)[C@H]([C@@H]1CCCCN1)c2ccccc2", "COC(=O)[C@@H]([C@H]1CCCCN1)c2ccccc2", "DAT", 5.0),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Loss Functions Test")
    print("=" * 60)

    # Test focal loss
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))
    labels[0] = -1  # Test ignore

    loss = focal(logits, labels)
    print(f"Focal loss: {loss.item():.4f}")

    # Test multi-task loss
    mt_loss = MultiTaskLoss(loss_fn='focal', learn_weights=True)

    predictions = {
        'DAT': torch.randn(5, 3),
        'NET': torch.randn(5, 3),
        'SERT': torch.randn(5, 3),
    }
    targets = {
        'DAT': torch.randint(0, 3, (5,)),
        'NET': torch.randint(0, 3, (5,)),
        'SERT': torch.randint(0, 3, (5,)),
    }

    losses = mt_loss(predictions, targets)
    print(f"\nMulti-task losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Test class weight computation
    test_labels = torch.tensor([0, 0, 0, 1, 2, 2])
    weights = compute_class_weights(test_labels, num_classes=3)
    print(f"\nClass weights for {test_labels.tolist()}: {weights.tolist()}")
