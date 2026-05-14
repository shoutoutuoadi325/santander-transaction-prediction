"""Evaluation helpers shared across notebooks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the standard binary-classification metric bundle.

    Returns a dict that's easy to ``pd.DataFrame`` or stash in a results table.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Return a DataFrame of metrics over a grid of thresholds.

    Default grid is 0.05..0.95 in 0.05 steps, which is fine-grained enough to
    pick a sensible operating point for class-imbalanced problems.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = [evaluate_binary(y_true, y_prob, threshold=float(t)) for t in thresholds]
    return pd.DataFrame(rows)


def best_threshold_by_f1(threshold_table: pd.DataFrame) -> Tuple[float, float]:
    """Return ``(threshold, f1)`` for the row with the highest F1 score."""
    row = threshold_table.loc[threshold_table["f1"].idxmax()]
    return float(row["threshold"]), float(row["f1"])


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute ROC curve points + AUC in one shot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(y_true, y_prob))}


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute PR curve points + AP in one shot."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision,
        "recall": recall,
        "ap": float(average_precision_score(y_true, y_prob)),
    }


def summarise_models(
    results: Dict[str, Dict[str, float]],
    metric_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Stack a dict of ``{model_name: metric_dict}`` into a comparison table."""
    if metric_order is None:
        metric_order = ["roc_auc", "pr_auc", "f1", "recall", "precision"]
    df = pd.DataFrame(results).T
    keep = [m for m in metric_order if m in df.columns]
    return df[keep + [c for c in df.columns if c not in keep]]
