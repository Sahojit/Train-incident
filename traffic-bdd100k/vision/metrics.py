"""
Multi-label evaluation metrics for BDD100K classifier.

Implements:
  - Per-label F1 score
  - Exact match ratio  (subset accuracy)
  - Hamming loss
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


LABEL_NAMES = ["rain", "night", "clear", "daytime"]


# ── Core helpers ──────────────────────────────────────────────────────────────

def binarize(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply sigmoid then threshold to convert raw logits → binary predictions."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return (probs >= threshold).astype(np.int32)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.int32)


# ── Metric functions ──────────────────────────────────────────────────────────

def per_label_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute binary F1 independently for each label column.

    Args:
        y_true: (N, L) ground-truth binary array.
        y_pred: (N, L) predicted binary array.

    Returns:
        dict mapping label name → F1 score in [0, 1].
    """
    if label_names is None:
        label_names = LABEL_NAMES
    assert y_true.shape == y_pred.shape, "Shape mismatch"

    scores: Dict[str, float] = {}
    for i, name in enumerate(label_names):
        tp = ((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()
        fp = ((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        scores[name] = float(f1)

    return scores


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of samples where the *entire* label vector is predicted correctly
    (also called subset accuracy).
    """
    return float(np.all(y_true == y_pred, axis=1).mean())


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of individual label slots that are incorrectly predicted.
    Lower is better (0.0 = perfect).
    """
    return float(np.mean(y_true != y_pred))


# ── Aggregated evaluator ──────────────────────────────────────────────────────

class MetricTracker:
    """Accumulates batched predictions and computes all metrics at epoch end."""

    def __init__(self):
        self._logits: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._logits.append(logits.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        all_logits  = torch.cat(self._logits,  dim=0)
        all_targets = torch.cat(self._targets, dim=0)

        y_pred = binarize(all_logits, threshold=threshold)
        y_true = to_numpy(all_targets)

        f1_scores = per_label_f1(y_true, y_pred)
        macro_f1  = float(np.mean(list(f1_scores.values())))

        return {
            **{f"f1_{k}": v for k, v in f1_scores.items()},
            "macro_f1":    macro_f1,
            "exact_match": exact_match_ratio(y_true, y_pred),
            "hamming_loss": hamming_loss(y_true, y_pred),
        }

    def reset(self) -> None:
        self._logits.clear()
        self._targets.clear()


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    header = f"{prefix} " if prefix else ""
    print(f"\n{header}Metrics")
    print(f"  Macro F1      : {metrics['macro_f1']:.4f}")
    print(f"  Exact match   : {metrics['exact_match']:.4f}")
    print(f"  Hamming loss  : {metrics['hamming_loss']:.4f}")
    for name in LABEL_NAMES:
        print(f"  F1 [{name:>8s}]: {metrics.get(f'f1_{name}', 0):.4f}")
