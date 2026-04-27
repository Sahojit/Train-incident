from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss

LABELS = ["rain", "night", "congestion", "clear"]


def binarize(probs: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).cpu().numpy().astype(int)


def compute_metrics(
    all_preds: torch.Tensor,
    all_targets: torch.Tensor,
    threshold: float = 0.5,
    label_names: List[str] = LABELS,
) -> Dict[str, float]:
    probs = torch.sigmoid(all_preds) if all_preds.max() > 1 else all_preds
    preds_bin = binarize(probs, threshold)
    targets_bin = all_targets.cpu().numpy().astype(int)

    metrics: Dict[str, float] = {}

    per_label_f1 = f1_score(targets_bin, preds_bin, average=None, zero_division=0)
    for name, score in zip(label_names, per_label_f1):
        metrics[f"f1_{name}"] = round(float(score), 4)

    metrics["f1_macro"] = round(float(f1_score(targets_bin, preds_bin, average="macro", zero_division=0)), 4)

    exact_match = np.all(preds_bin == targets_bin, axis=1).mean()
    metrics["exact_match_ratio"] = round(float(exact_match), 4)

    metrics["hamming_loss"] = round(float(hamming_loss(targets_bin, preds_bin)), 4)

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = ["=" * 40, "Evaluation Metrics", "=" * 40]
    for key, val in metrics.items():
        lines.append(f"  {key:<25}: {val:.4f}")
    lines.append("=" * 40)
    return "\n".join(lines)


if __name__ == "__main__":
    preds = torch.randn(32, 4)
    targets = torch.randint(0, 2, (32, 4)).float()
    m = compute_metrics(preds, targets)
    print(format_metrics(m))
