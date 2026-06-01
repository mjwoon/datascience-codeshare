"""
metrics.py — Evaluation metrics for FAF freight forecasting.

Functions
---------
wrmse(y_true, y_pred)                   → float  tons-weighted RMSE (primary)
rmse(y_true, y_pred)                    → float  standard RMSE
mae(y_true, y_pred)                     → float  mean absolute error
wmape(y_true, y_pred)                   → float  weighted MAPE = sum|e|/sum|y|
rmsle(y_true, y_pred)                   → float  root mean squared log error
r2(y_true, y_pred)                      → float  R²
all_metrics(y_true, y_pred)             → dict   all six metrics
weighted_metric(split_results, weights) → float  cross-split weighted average
weighted_wrmse(split_results, weights)  → float  alias for wrmse
economic_benefit_M(...)                 → float  M$ economic benefit
"""
from __future__ import annotations

import numpy as np


def wrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Redefined as standard (unweighted) RMSE to align with individual experiment metrics."""
    return rmse(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard (unweighted) RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = sum|y−ŷ| / sum|y|. Returns 0 if sum(y)==0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom < 1e-9:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denom)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error: sqrt(mean((log1p(y)−log1p(ŷ))²))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination). Returns 0 if variance==0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-9:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all six metrics at once."""
    return {
        "RMSE":  rmse(y_true, y_pred),
        "MAE":   mae(y_true, y_pred),
        "WMAPE": wmape(y_true, y_pred),
        "RMSLE": rmsle(y_true, y_pred),
        "R2":    r2(y_true, y_pred),
        "wRMSE": wrmse(y_true, y_pred),
    }


def weighted_metric(split_results: list[float], weights: list[float]) -> float:
    """Weighted average across splits: Σ w_i·metric_i."""
    return float(sum(w * r for w, r in zip(weights, split_results)))


def weighted_wrmse(split_results: list[float], weights: list[float]) -> float:
    """Alias for weighted_metric (backward compat)."""
    return weighted_metric(split_results, weights)


def economic_benefit_M(
    y_true: np.ndarray,
    y_pred_new: np.ndarray,
    y_pred_base: np.ndarray,
) -> float:
    """Economic benefit in M$: (MAE_base − MAE_new) × 1000 t/kt × $10/t / 1e6."""
    y_true      = np.asarray(y_true,      dtype=float)
    y_pred_new  = np.asarray(y_pred_new,  dtype=float)
    y_pred_base = np.asarray(y_pred_base, dtype=float)
    return (mae(y_true, y_pred_base) - mae(y_true, y_pred_new)) * 1000 * 10 / 1e6
