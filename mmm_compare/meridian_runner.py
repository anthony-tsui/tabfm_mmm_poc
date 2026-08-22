"""Meridian baseline with deliverable extraction + Meridian-style proxy fallback."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mmm_compare.data import MMMDataset
from mmm_compare.metrics import contribution_recovery_metrics, regression_metrics
from mmm_compare.tabfm_runner import ModelResult, _true_holdout_contributions

logger = logging.getLogger(__name__)


def geometric_adstock(x: np.ndarray, retention: float = 0.5) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + (retention * out[t - 1] if t else 0.0)
    return out


def hill_saturation(x: np.ndarray, ec50: float = 1.0, slope: float = 1.5) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return (x**slope) / (x**slope + ec50**slope + 1e-12)


def _media_design_matrix(df: pd.DataFrame, channel_keys: list[str]) -> tuple[pd.DataFrame, list[str]]:
    parts: dict[str, np.ndarray] = {}
    media_cols: list[str] = []
    for key in channel_keys:
        spend_col = f"{key}_spend"
        if spend_col not in df.columns:
            continue
        spend = df[spend_col].to_numpy(dtype=float)
        scale = np.median(spend) + 1e-9
        transformed = hill_saturation(geometric_adstock(spend / scale, retention=0.4))
        name = f"{key}_media_transform"
        parts[name] = transformed
        media_cols.append(name)
    for c in df.columns:
        if c.endswith("_control") or c in ("Promo", "promo_control") or c.startswith("Organic_"):
            if pd.api.types.is_numeric_dtype(df[c]):
                parts[c] = df[c].to_numpy(dtype=float)
    X = pd.DataFrame(parts, index=df.index)
    return X, media_cols


def run_meridian_style(data: MMMDataset) -> ModelResult:
    """Lightweight media-mix style proxy (adstock + Hill + Ridge) — not full Meridian."""
    X_all, media_cols = _media_design_matrix(data.frame, data.channel_keys)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_all.loc[data.train_idx], data.y_train)
    pred = model.predict(X_all.loc[data.test_idx])
    metrics = regression_metrics(data.y_test, pred)

    contrib: dict[str, float] = {}
    coef_map = dict(zip(X_all.columns, model.coef_))
    X_test = X_all.loc[data.test_idx]
    for key in data.channel_keys:
        col = f"{key}_media_transform"
        if col in coef_map:
            contrib[key] = float(np.mean(np.maximum(coef_map[col] * X_test[col].to_numpy(), 0.0)))

    true_c = _true_holdout_contributions(data)
    c_metrics = contribution_recovery_metrics(true_c, contrib) if true_c else {}

    # Proxy "ROI-like" = mean positive coef contribution / mean spend (NOT Meridian ROI)
    proxy_roi = {}
    for key, cval in contrib.items():
        spend_col = f"{key}_spend"
        if spend_col in data.frame.columns:
            spend = float(data.frame.loc[data.test_idx, spend_col].mean())
            proxy_roi[key] = float(cval / spend) if spend > 0 else float("nan")

    return ModelResult(
        name="Meridian-style baseline",
        mode="meridian_style",
        y_pred_test=np.asarray(pred, dtype=float),
        metrics=metrics,
        contribution_pred=contrib,
        contribution_metrics=c_metrics,
        extras={
            "note": (
                "Proxy: geometric adstock + Hill saturation + Ridge. "
                "Not full Google Meridian Bayesian MCMC. Proxy ROI is NOT Meridian ROI."
            ),
            "media_cols": media_cols,
            "deliverables": {
                "expected_outcome": "proxy holdout predictions only",
                "channel_contribution": contrib,
                "roi_by_channel_proxy": proxy_roi,
                "response_curves": None,
                "budget_optimization": None,
                "geo_insights": None,
            },
        },
    )


def _posterior_mean(tensor) -> np.ndarray:
    arr = np.asarray(tensor, dtype=float)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr


def _xr_to_records(ds_or_da) -> list[dict[str, Any]] | dict[str, Any] | None:
    try:
        import xarray as xr

        if isinstance(ds_or_da, xr.Dataset):
            return ds_or_da.to_dataframe().reset_index().to_dict(orient="records")
        if isinstance(ds_or_da, xr.DataArray):
            return ds_or_da.to_dataframe(name=ds_or_da.name or "value").reset_index().to_dict(
                orient="records"
            )
    except Exception as exc:
        logger.info("xarray convert skipped: %s", exc)
    try:
        return {"raw_shape": list(np.asarray(ds_or_da).shape)}
    except Exception:
        return None


def _build_meridian_input(data: MMMDataset):
    from meridian.data import data_frame_input_data_builder

    df = data.frame.copy()
    if "revenue_per_conversion" not in df.columns:
        df["revenue_per_conversion"] = 1.0

    channels, media_cols, spend_cols = [], [], []
    for key in data.channel_keys:
        imp, sp = f"{key}_impression", f"{key}_spend"
        if imp in df.columns and sp in df.columns:
            channels.append(key)
            media_cols.append(imp)
            spend_cols.append(sp)

    control_cols = [c for c in df.columns if c.endswith("_control")]
    non_media = [c for c in ("Promo",) if c in df.columns]
    organic_cols = [c for c in df.columns if c.startswith("Organic_") and c.endswith("_impression")]
    organic_channels = [c.replace("_impression", "") for c in organic_cols]

    kpi_col = data.kpi_col
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue",
        default_kpi_column=kpi_col,
        default_revenue_per_kpi_column="revenue_per_conversion",
    )
    builder = builder.with_kpi(df).with_revenue_per_kpi(df)
    if "population" in df.columns and data.is_geo:
        builder = builder.with_population(df)
    if control_cols:
        builder = builder.with_controls(df, control_cols=control_cols)
    builder = builder.with_media(
        df,
        media_cols=media_cols,
        media_spend_cols=spend_cols,
        media_channels=channels,
    )
    if organic_cols:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=organic_cols,
            organic_media_channels=organic_channels,
        )
    if non_media:
        builder = builder.with_non_media_treatments(df, non_media_treatment_cols=non_media)
    return builder.build(), channels


def _extract_deliverables(ana, channels: list[str]) -> dict[str, Any]:
    """Pull Meridian Analyzer artifacts; best-effort for PoC."""
    out: dict[str, Any] = {
        "expected_outcome": None,
        "predictive_accuracy": None,
        "channel_contribution": {},
        "roi_by_channel": {},
        "summary_metrics": None,
        "response_curves": None,
        "budget_optimization": None,
        "geo_insights": None,
        "extraction_notes": [],
    }

    try:
        pa = ana.predictive_accuracy(use_kpi=True)
        out["predictive_accuracy"] = _xr_to_records(pa)
    except Exception as exc:
        out["extraction_notes"].append(f"predictive_accuracy: {exc}")

    try:
        expected = ana.expected_outcome(aggregate_times=False, aggregate_geos=True, use_kpi=True)
        ts = _posterior_mean(expected)
        out["expected_outcome"] = {
            "n_times": int(len(ts)),
            "mean": float(np.mean(ts)),
            "time_series_head": ts[:5].tolist(),
            "time_series_tail": ts[-5:].tolist(),
        }
        out["_expected_time_series"] = ts
    except Exception as exc:
        out["extraction_notes"].append(f"expected_outcome: {exc}")

    try:
        effects = ana.incremental_outcome(aggregate_times=True, aggregate_geos=True, use_kpi=True)
        e = _posterior_mean(effects)
        if len(e) >= len(channels):
            out["channel_contribution"] = {
                ch: float(max(v, 0.0)) for ch, v in zip(channels, e[: len(channels)])
            }
    except Exception as exc:
        out["extraction_notes"].append(f"incremental_outcome: {exc}")

    try:
        roi_t = ana.roi(aggregate_geos=True, use_kpi=True)
        r = _posterior_mean(roi_t)
        if len(r) >= len(channels):
            out["roi_by_channel"] = {ch: float(v) for ch, v in zip(channels, r[: len(channels)])}
    except Exception as exc:
        out["extraction_notes"].append(f"roi: {exc}")

    try:
        sm = ana.summary_metrics(aggregate_geos=True, aggregate_times=True, use_kpi=True)
        out["summary_metrics"] = _xr_to_records(sm)
    except Exception as exc:
        out["extraction_notes"].append(f"summary_metrics: {exc}")

    try:
        # Coarse response curve grid for PoC
        rc = ana.response_curves(
            spend_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5],
            use_kpi=True,
        )
        # Keep a compact preview, not the full posterior grid
        preview = _xr_to_records(rc)
        if isinstance(preview, list) and len(preview) > 40:
            out["response_curves"] = {"n_rows": len(preview), "preview": preview[:20]}
        else:
            out["response_curves"] = preview
    except Exception as exc:
        out["extraction_notes"].append(f"response_curves: {exc}")

    out["budget_optimization"] = {
        "available_in_meridian": True,
        "extracted_in_this_poc": False,
        "note": "Use meridian.analysis.optimizer in a full Meridian workflow; skipped here for PoC runtime.",
    }
    return out


def _try_meridian(data: MMMDataset, *, n_keep: int = 50) -> ModelResult | None:
    try:
        from meridian import constants
        from meridian.analysis import analyzer
        from meridian.model import model, prior_distribution, spec
        import tensorflow_probability as tfp
    except Exception as exc:
        logger.info("google-meridian not available (%s)", exc)
        return None

    if data.is_geo:
        # Full 40×156 geo MCMC is too heavy for typical CPU PoC VMs.
        n_geos = int(data.frame["geo"].nunique()) if "geo" in data.frame.columns else 0
        n_times = int(data.frame["time"].nunique()) if "time" in data.frame.columns else 0
        if n_geos * n_times > 500:
            logger.warning(
                "Skipping full geo Meridian fit (%s geos × %s times). "
                "Use --dataset national (default) or --dataset geo-agg.",
                n_geos,
                n_times,
            )
            return None

    try:
        input_data, channels = _build_meridian_input(data)
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M)
        )
        mmm = model.Meridian(input_data=input_data, model_spec=spec.ModelSpec(prior=prior))
        mmm.sample_prior(50)
        mmm.sample_posterior(
            n_chains=1,
            n_adapt=80,
            n_burnin=40,
            n_keep=n_keep,
            seed=42,
        )
        ana = analyzer.Analyzer(mmm)
        deliverables = _extract_deliverables(ana, channels)

        ts = deliverables.pop("_expected_time_series", None)
        if ts is None or len(ts) == 0:
            logger.warning("Meridian expected_outcome unavailable; falling back")
            return None

        # Align holdout predictions to unique times for national; for geo panel,
        # expected_outcome is national-aggregated over geos → compare to national KPI sum.
        if data.is_geo:
            test_times = sorted(data.frame.loc[data.test_idx, "time"].unique())
            all_times = sorted(data.frame["time"].unique())
            actual = (
                data.frame[data.frame["time"].isin(test_times)]
                .groupby("time")[data.kpi_col]
                .sum()
                .reindex(test_times)
                .to_numpy(dtype=float)
            )
            idx = [all_times.index(t) for t in test_times]
            if max(idx) >= len(ts):
                logger.warning("Meridian geo time index out of range")
                return None
            pred = ts[idx]
            metrics = regression_metrics(actual, pred)
            y_pred = np.asarray(pred, dtype=float)
        else:
            if len(ts) != len(data.frame):
                logger.warning("Meridian time length mismatch (%s vs %s)", len(ts), len(data.frame))
                return None
            pred = ts[np.asarray(data.test_idx)]
            metrics = regression_metrics(data.y_test, pred)
            y_pred = np.asarray(pred, dtype=float)

        contrib = deliverables.get("channel_contribution") or {}
        true_c = _true_holdout_contributions(data)
        c_metrics = contribution_recovery_metrics(true_c, contrib) if true_c and contrib else {}

        if not data.is_geo:
            deliverables["geo_insights"] = {
                "model": "national",
                "note": "National model — no geo-level Meridian insights in this run.",
            }
        else:
            deliverables["geo_insights"] = {
                "model": "geo",
                "n_geos": int(data.frame["geo"].nunique()),
                "note": "Geo model fitted; detailed per-geo tables omitted in PoC payload.",
            }

        return ModelResult(
            name="Meridian",
            mode="meridian",
            y_pred_test=y_pred,
            metrics=metrics,
            contribution_pred={str(k): float(v) for k, v in contrib.items()},
            contribution_metrics=c_metrics,
            extras={
                "n_keep": n_keep,
                "n_chains": 1,
                "note": (
                    "Minimal Meridian MCMC PoC. Holdout KPI uses in-sample expected_outcome "
                    "on later weeks (fit on full series)."
                ),
                "deliverables": deliverables,
            },
        )
    except Exception as exc:
        logger.warning("Meridian fit failed (%s)", exc)
        return None


def run_meridian_baseline(
    data: MMMDataset,
    *,
    prefer_real: bool = True,
    dry_run: bool = False,
    n_keep: int = 50,
) -> ModelResult:
    if dry_run:
        result = run_meridian_style(data)
        result.extras["dry_run"] = True
        return result
    if prefer_real:
        real = _try_meridian(data, n_keep=n_keep)
        if real is not None:
            return real
    return run_meridian_style(data)
