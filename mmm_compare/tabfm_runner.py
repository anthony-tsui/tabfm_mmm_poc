"""TabFM regression path with dry-run / mock fallback.

Default path scores holdout KPI only. Leave-one-channel ablation is opt-in via
`--ablation-proxy` and is NOT Meridian-equivalent / not causal.
"""

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
    """Leave-one-channel-out prediction delta — NOT Meridian incremental_outcome."""
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


def _tabfm_deliverables(*, ablation: dict[str, float] | None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "predictive_kpi": "holdout KPI predictions via TabFMRegressor (only Yes vs Meridian)",
        "channel_contribution": None,
        "roi_by_channel": None,
        "response_curves": None,
        "budget_optimization": None,
        "geo_insights": None,
        "honesty": (
            "TabFM is a tabular ICL regressor, not an MMM. Predictive KPI is the only "
            "comparable deliverable. Contribution / ROI / curves / budget / geo decisioning "
            "are Meridian-only product surface (hard No for TabFM)."
        ),
    }
    if ablation is not None:
        d["ablation_proxy_NOT_meridian_equivalent"] = {
            "warning": (
                "NOT Meridian-equivalent and not causal. Opt-in only via --ablation-proxy. "
                "Do not treat as channel contribution / incremental_outcome."
            ),
            "method": "leave-one-channel ablation on predictions",
            "values": ablation,
        }
    return d


def _finish(
    *,
    mode: str,
    pred: np.ndarray,
    metrics: dict[str, float],
    data: MMMDataset,
    predict_fn,
    ablation_proxy: bool,
    extras: dict[str, Any],
) -> ModelResult:
    contrib: dict[str, float] = {}
    c_metrics: dict[str, float] = {}
    ablation = None
    if ablation_proxy:
        ablation = _ablation_contributions(predict_fn, data.X_test, data.channel_keys, pred)
        # Kept off the main contribution_pred headline unless explicitly requested;
        # still available under extras for inspection.
        true_c = _true_holdout_contributions(data)
        c_metrics = contribution_recovery_metrics(true_c, ablation) if true_c else {}
        extras = {
            **extras,
            "ablation_proxy_enabled": True,
            "ablation_proxy_warning": (
                "NOT Meridian-equivalent / not causal — excluded from headline metrics table"
            ),
        }
    extras["deliverables"] = _tabfm_deliverables(ablation=ablation)
    return ModelResult(
        name="TabFM",
        mode=mode,
        y_pred_test=np.asarray(pred, dtype=float),
        metrics=metrics,
        contribution_pred=contrib,  # empty by default — do not mirror Meridian contrib
        contribution_metrics=c_metrics if ablation_proxy else {},
        extras=extras,
    )


def _mock_tabfm(data: MMMDataset, *, ablation_proxy: bool = False) -> ModelResult:
    model = Ridge(alpha=1.0)
    model.fit(data.X_train, data.y_train)
    pred = model.predict(data.X_test)
    metrics = regression_metrics(data.y_test, pred)
    return _finish(
        mode="mock",
        pred=pred,
        metrics=metrics,
        data=data,
        predict_fn=model.predict,
        ablation_proxy=ablation_proxy,
        extras={"note": "Dry-run mock (Ridge). TabFM weights not used."},
    )


def run_tabfm(
    data: MMMDataset,
    *,
    dry_run: bool = False,
    force_mock: bool = False,
    ablation_proxy: bool = False,
) -> ModelResult:
    if dry_run or force_mock:
        logger.info("TabFM dry-run/mock path enabled")
        return _mock_tabfm(data, ablation_proxy=ablation_proxy)

    try:
        from tabfm import TabFMRegressor
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    except Exception as exc:  # pragma: no cover
        logger.warning("TabFM import failed (%s); falling back to mock", exc)
        result = _mock_tabfm(data, ablation_proxy=ablation_proxy)
        result.extras["fallback_reason"] = f"import_error: {exc}"
        return result

    try:
        backbone = tabfm_v1_0_0.load(model_type="regression")
        reg = TabFMRegressor(model=backbone)
        reg.fit(data.X_train, data.y_train.to_numpy())
        pred = np.asarray(reg.predict(data.X_test), dtype=float)
        metrics = regression_metrics(data.y_test, pred)
        return _finish(
            mode="tabfm",
            pred=pred,
            metrics=metrics,
            data=data,
            predict_fn=reg.predict,
            ablation_proxy=ablation_proxy,
            extras={"backend": "pytorch", "weights": "google/tabfm-1.0.0-pytorch"},
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("TabFM inference failed (%s); falling back to mock", exc)
        result = _mock_tabfm(data, ablation_proxy=ablation_proxy)
        result.extras["fallback_reason"] = f"runtime_error: {exc}"
        return result
