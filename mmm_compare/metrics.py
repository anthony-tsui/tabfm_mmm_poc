"""Shared evaluation metrics for holdout KPI prediction and contribution recovery."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape}


def contribution_recovery_metrics(
    true_contrib: Mapping[str, float] | pd.Series,
    pred_contrib: Mapping[str, float] | pd.Series,
) -> dict[str, float]:
    """Compare channel contribution shares (or totals) via MAE and correlation."""
    keys = sorted(set(true_contrib.keys()) & set(pred_contrib.keys()))
    if not keys:
        return {}
    t = np.array([float(true_contrib[k]) for k in keys], dtype=float)
    p = np.array([float(pred_contrib[k]) for k in keys], dtype=float)
    t_share = t / (t.sum() + 1e-12)
    p_share = p / (p.sum() + 1e-12)
    corr = float(np.corrcoef(t, p)[0, 1]) if len(keys) >= 2 else float("nan")
    share_corr = float(np.corrcoef(t_share, p_share)[0, 1]) if len(keys) >= 2 else float("nan")
    return {
        "contrib_mae": float(np.mean(np.abs(t - p))),
        "contrib_share_mae": float(np.mean(np.abs(t_share - p_share))),
        "contrib_corr": corr,
        "contrib_share_corr": share_corr,
    }


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model")
