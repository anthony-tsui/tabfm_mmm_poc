"""TabFM regression path with dry-run / mock fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mmm_compare.data import MMMDataset
from mmm_compare.metrics import contribution_recovery_metrics, regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    mode: str  # "tabfm" | "mock" | "meridian" | "meridian_style"
    y_pred_test: np.ndarray
    metrics: dict[str, float]
    contribution_pred: dict[str, float] = field(default_factory=dict)
    contribution_metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _ablation_contributions(
    predict_fn,
    X_test: pd.DataFrame,
    channel_keys: list[str],
    y_pred_base: np.ndarray,
) -> dict[str, float]:
    """Approximate channel contributions via leave-one-channel-out prediction delta."""
    out: dict[str, float] = {}
    for key in channel_keys:
        cols = [c for c in X_test.columns if c.startswith(f"{key}_")]
        if not cols:
            continue
        X0 = X_test.copy()
        X0[cols] = 0.0
        y0 = np.asarray(predict_fn(X0), dtype=float)
        out[key] = float(np.mean(np.maximum(y_pred_base - y0, 0.0)))
    return out


def _true_holdout_contributions(data: MMMDataset) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in data.contribution_cols:
        key = col.replace("contribution_", "", 1)
        out[key] = float(data.frame.loc[data.test_idx, col].mean())
    return out


def _tabfm_deliverables(contrib: dict[str, float]) -> dict[str, Any]:
    return {
        "expected_outcome": "holdout KPI predictions via TabFMRegressor (predictive only)",
        "channel_contribution": {
            "method": "leave-one-channel ablation (hacky proxy)",
            "values": contrib,
            "meridian_equivalent": False,
        },
        "roi_by_channel": None,
        "response_curves": None,
        "budget_optimization": None,
        "geo_insights": (
            "Row-level geo-week prediction possible if geo panel features are supplied; "
            "no Meridian hierarchical geo posteriors."
        ),
        "honesty": (
            "TabFM is a tabular ICL regressor, not an MMM. Only predictive KPI is a Yes; "
            "other Meridian deliverables are Partial/No — see deliverables matrix."
        ),
    }


def _mock_tabfm(data: MMMDataset) -> ModelResult:
    """Ridge on context rows — preserves metrics schema when TabFM weights are unavailable."""
    model = Ridge(alpha=1.0)
    model.fit(data.X_train, data.y_train)
    pred = model.predict(data.X_test)
    metrics = regression_metrics(data.y_test, pred)
    contrib = _ablation_contributions(model.predict, data.X_test, data.channel_keys, pred)
    true_c = _true_holdout_contributions(data)
    c_metrics = contribution_recovery_metrics(true_c, contrib) if true_c else {}
    return ModelResult(
        name="TabFM",
        mode="mock",
        y_pred_test=np.asarray(pred, dtype=float),
        metrics=metrics,
        contribution_pred=contrib,
        contribution_metrics=c_metrics,
        extras={
            "note": "Dry-run mock (Ridge). TabFM weights not used.",
            "deliverables": _tabfm_deliverables(contrib),
        },
    )


def run_tabfm(data: MMMDataset, *, dry_run: bool = False, force_mock: bool = False) -> ModelResult:
    if dry_run or force_mock:
        logger.info("TabFM dry-run/mock path enabled")
        return _mock_tabfm(data)

    try:
        from tabfm import TabFMRegressor
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    except Exception as exc:  # pragma: no cover - env dependent
        logger.warning("TabFM import failed (%s); falling back to mock", exc)
        result = _mock_tabfm(data)
        result.extras["fallback_reason"] = f"import_error: {exc}"
        return result

    try:
        backbone = tabfm_v1_0_0.load(model_type="regression")
        reg = TabFMRegressor(model=backbone)
        reg.fit(data.X_train, data.y_train.to_numpy())
        pred = np.asarray(reg.predict(data.X_test), dtype=float)
        metrics = regression_metrics(data.y_test, pred)
        contrib = _ablation_contributions(reg.predict, data.X_test, data.channel_keys, pred)
        true_c = _true_holdout_contributions(data)
        c_metrics = contribution_recovery_metrics(true_c, contrib) if true_c else {}
        return ModelResult(
            name="TabFM",
            mode="tabfm",
            y_pred_test=pred,
            metrics=metrics,
            contribution_pred=contrib,
            contribution_metrics=c_metrics,
            extras={
                "backend": "pytorch",
                "weights": "google/tabfm-1.0.0-pytorch",
                "deliverables": _tabfm_deliverables(contrib),
            },
        )
    except Exception as exc:  # pragma: no cover - weights / runtime
        logger.warning("TabFM inference failed (%s); falling back to mock", exc)
        result = _mock_tabfm(data)
        result.extras["fallback_reason"] = f"runtime_error: {exc}"
        return result
