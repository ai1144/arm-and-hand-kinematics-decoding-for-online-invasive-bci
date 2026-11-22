from __future__ import annotations
"""Core offline-decoding metrics used across the VM suite.

These helpers are intentionally small and dependency-light so every decoder
family is evaluated with the same R2, RMSE, correlation, and jitter definitions.
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error across all time bins and output dimensions."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average per-dimension R2, ignoring dimensions with almost no variance."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    valid = ss_tot > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(1.0 - ss_res[valid] / ss_tot[valid]))


def per_dim_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return one R2 value for each qpos dimension."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = ss_tot > 1e-12
    out[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return out


def mean_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Pearson correlation across dimensions with nonzero variance."""
    corrs = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        sa = np.std(a)
        sb = np.std(b)
        if sa <= 1e-12 or sb <= 1e-12:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def jitter(y_pred: np.ndarray) -> float:
    """Second-difference magnitude; higher values mean shakier decoded motion."""
    if y_pred.shape[0] < 3:
        return 0.0
    acc = np.diff(y_pred, n=2, axis=0)
    return float(np.mean(np.abs(acc)))


def drift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Difference between prediction error at trial end and prediction error at start."""
    return float(np.linalg.norm((y_pred[-1] - y_true[-1]) - (y_pred[0] - y_true[0])))


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the shared metric dictionary used by all decoder families."""
    r2j = per_dim_r2(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "CC": mean_corr(y_true, y_pred),
        "med_R2": float(np.nanmedian(r2j)),
        "worst_R2": float(np.nanmin(r2j)),
        "jitter": jitter(y_pred),
        "drift": drift(y_true, y_pred),
        "nan_count": int(np.isnan(y_pred).sum()),
        "inf_count": int(np.isinf(y_pred).sum()),
    }
