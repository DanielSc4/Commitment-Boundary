"""Classification metrics, random baselines, and threshold sweep utilities.

Shared by train_probe.py, train_attn_probe.py, and probe_compression_exps.py.
"""

from typing import Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(preds, y, logits=None):
    """Compute accuracy, precision, recall, F1, optionally AUROC."""
    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()

    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    auroc = None
    if logits is not None:
        try:
            from sklearn.metrics import roc_auc_score
            _logits_np = logits.cpu().numpy()
            _y_np = y.cpu().numpy()
            _nan_mask = ~__import__("numpy").isnan(_logits_np)
            if not _nan_mask.all():
                print(f"WARNING: {(~_nan_mask).sum()} NaN logit(s) excluded from AUROC")
                _logits_np = _logits_np[_nan_mask]
                _y_np = _y_np[_nan_mask]
            auroc = float(roc_auc_score(_y_np, _logits_np))
        except ImportError:
            print("WARNING: sklearn not found, AUROC not computed")
        except ValueError as e:
            print(f"WARNING: AUROC computation failed: {e}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Random baselines
# ---------------------------------------------------------------------------


def compute_random_baselines(y: torch.Tensor, seed: int = 0) -> dict:
    """Compute two uninformed baselines on binary labels y.

    - always_negative: predict all 0s (F1=0 regardless of threshold)
    - random_prior:    predict 1 with probability = pos_rate (seeded)

    Returns dict mapping baseline name -> metrics dict.
    """
    y = y.float()
    n = len(y)
    pos_rate = y.mean().item() if n > 0 else 0.0

    preds_neg = torch.zeros(n)
    rng = torch.Generator().manual_seed(seed)
    preds_rand = (torch.rand(n, generator=rng) < pos_rate).float()

    return {
        "always_negative": compute_classification_metrics(preds_neg, y),
        "random_prior":    compute_classification_metrics(preds_rand, y),
    }


def print_baselines(baselines: dict) -> None:
    print("  Baseline            Acc     P       R       F1")
    for name, m in baselines.items():
        print(f"  {name:<20s}"
              f"  {m['accuracy']:.4f}  {m['precision']:.4f}  "
              f"{m['recall']:.4f}  {m['f1']:.4f}")


# ---------------------------------------------------------------------------
# Threshold sweep on pre-collected logits
# ---------------------------------------------------------------------------


def sweep_threshold_curve(
    logits: torch.Tensor,
    labels: torch.Tensor,
    low: float = -3.0,
    high: float = 5.0,
    step: float = 0.1,
) -> dict:
    """Sweep classification thresholds on pre-collected logits.

    Returns dict with:
        pr_curve: list of {threshold, precision, recall, f1}
        best_f1: {threshold, precision, recall, f1}
        best_precision_at_recall_50: {threshold, precision, recall, f1}
        best_precision_at_recall_30: {threshold, precision, recall, f1}
    """
    labels = labels.float()

    thresholds = [round(low + i * step, 1) for i in range(int((high - low) / step) + 1)]
    pr_curve = []

    for th in thresholds:
        preds = (logits > th).float()
        tp = int(((preds == 1) & (labels == 1)).sum().item())
        fp = int(((preds == 1) & (labels == 0)).sum().item())
        fn = int(((preds == 0) & (labels == 1)).sum().item())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        pr_curve.append({"threshold": th, "precision": p, "recall": r, "f1": f1})

    # Best F1
    best_f1 = max(pr_curve, key=lambda x: x["f1"])

    # Best precision at recall >= 0.5
    candidates_50 = [x for x in pr_curve if x["recall"] >= 0.5]
    best_p_at_r50 = max(candidates_50, key=lambda x: x["precision"]) if candidates_50 else best_f1

    # Best precision at recall >= 0.3
    candidates_30 = [x for x in pr_curve if x["recall"] >= 0.3]
    best_p_at_r30 = max(candidates_30, key=lambda x: x["precision"]) if candidates_30 else best_f1

    return {
        "pr_curve": pr_curve,
        "best_f1": best_f1,
        "best_precision_at_recall_50": best_p_at_r50,
        "best_precision_at_recall_30": best_p_at_r30,
    }
