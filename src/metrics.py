"""Local challenge metrics: Micro-F1 (aspect presence), sentiment accuracy, overall."""

from __future__ import annotations

import numpy as np
import torch


ASPECT_COLS = ("giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam")


def labels_to_numpy(y: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    return np.asarray(y, dtype=np.int64)


def micro_f1_aspect_presence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Micro-F1: positive iff label > 0.
    Flattens 6 aspects so micro = global TP/FP/FN over all (sample, aspect) pairs.
    """
    assert y_true.shape == y_pred.shape
    yt = (y_true > 0).astype(np.int64).ravel()
    yp = (y_pred > 0).astype(np.int64).ravel()
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    if tp == 0 and (fp + fn) == 0:
        return 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def sentiment_accuracy_masked(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy on sentiment classes 1..5 only where y_true > 0."""
    assert y_true.shape == y_pred.shape
    mask = y_true > 0
    if not mask.any():
        return 1.0
    correct = (y_true[mask] == y_pred[mask]).astype(np.float64)
    return float(correct.mean())


def overall_score(micro_f1: float, sentiment_acc: float, w_f1: float = 0.7, w_sent: float = 0.3) -> float:
    return w_f1 * micro_f1 + w_sent * sentiment_acc


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mf1 = micro_f1_aspect_presence(y_true, y_pred)
    sa = sentiment_accuracy_masked(y_true, y_pred)
    return {
        "micro_f1": mf1,
        "sentiment_accuracy": sa,
        "overall_score": overall_score(mf1, sa),
    }
