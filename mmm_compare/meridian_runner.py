"""Meridian baseline with shared holdout_id OOS scoring + Meridian-style proxy."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mmm_compare.data import MMMDataset
from mmm_compare.metrics import regression_metrics
from mmm_compare.tabfm_runner import ModelResult

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
    return pd.DataFrame(parts, index=df.index), media_cols


def run_meridian_style(data: MMMDataset) -> ModelResult:
    """Adstock + Hill + Ridge on the same train/holdout split (not Bayesian Meridian)."""
    X_all, media_cols = _media_design_matrix(data.frame, data.channel_keys)
    # Use full Meridian train weeks (not TabFM context cap) for the style proxy.
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_all.loc[data.train_idx], data.frame.loc[data.train_idx, data.kpi_col])
    pred = model.predict(X_all.loc[data.test_idx])
    y_test = data.frame.loc[data.test_idx, data.kpi_col]
    metrics = regression_metrics(y_test, pred)
    return ModelResult(
        name="Meridian-style baseline",
        mode="meridian_style",
        y_pred_test=np.asarray(pred, dtype=float),
        metrics=metrics,
        contribution_pred={},
        contribution_metrics={},
        extras={
            "note": (
                "Proxy: geometric adstock + Hill + Ridge on the shared time holdout. "
                "Not full Google Meridian MCMC. No Meridian ROI/contribution tables."
            ),
            "media_cols": media_cols,
            "holdout_times": data.holdout_times,
            "eval": "oos_holdout_same_as_tabfm",
            "deliverables": {
                "predictive_kpi": "holdout predictions from style proxy",
                "channel_contribution": None,
                "roi_by_channel": None,
                "response_curves": None,
                "budget_optimization": None,
                "mcmc_quality": "n/a (not Meridian)",
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

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type="non_revenue",
        default_kpi_column=data.kpi_col,
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
    """Meridian-only product surface (directional PoC MCMC — not decision-grade)."""
    out: dict[str, Any] = {
        "predictive_accuracy": None,
        "channel_contribution": {},
        "roi_by_channel": {},
        "summary_metrics": None,
        "response_curves": None,
        "budget_optimization": {
            "available_in_meridian": True,
            "extracted_in_this_poc": False,
            "note": "meridian.analysis.optimizer skipped for PoC runtime.",
        },
        "geo_insights": None,
        "mcmc_quality": "directional_poc_not_decision_grade",
        "extraction_notes": [],
    }
    try:
        pa = ana.predictive_accuracy(use_kpi=True)
        out["predictive_accuracy"] = _xr_to_records(pa)
    except Exception as exc:
        out["extraction_notes"].append(f"predictive_accuracy: {exc}")

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
        rc = ana.response_curves(spend_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5], use_kpi=True)
        preview = _xr_to_records(rc)
        if isinstance(preview, list) and len(preview) > 40:
            out["response_curves"] = {"n_rows": len(preview), "preview": preview[:20]}
        else:
            out["response_curves"] = preview
    except Exception as exc:
        out["extraction_notes"].append(f"response_curves: {exc}")

    return out


def _holdout_predictions_from_expected(
    data: MMMDataset,
    time_series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice expected_outcome to the shared holdout weeks; return (y_true, y_pred)."""
    if data.is_geo:
        test_times = data.holdout_times
        all_times = [str(t) for t in sorted(data.frame["time"].unique())]
        actual = (
            data.frame[data.frame["time"].astype(str).isin(test_times)]
            .groupby(data.frame["time"].astype(str))[data.kpi_col]
            .sum()
            .reindex(test_times)
            .to_numpy(dtype=float)
        )
        idx = [all_times.index(t) for t in test_times]
        if max(idx) >= len(time_series):
            raise ValueError("holdout time index out of range for Meridian expected_outcome")
        pred = time_series[idx]
        return actual, pred

    if len(time_series) != len(data.frame):
        raise ValueError(
            f"Meridian expected_outcome length {len(time_series)} != n_times {len(data.frame)}"
        )
    mask = data.holdout_id.astype(bool)
    if mask.ndim != 1 or len(mask) != len(data.frame):
        raise ValueError(f"national holdout_id shape {mask.shape} incompatible with frame")
    y_true = data.frame.loc[mask, data.kpi_col].to_numpy(dtype=float)
    y_pred = time_series[mask]
    return y_true, y_pred


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
        n_geos = int(data.frame["geo"].nunique()) if "geo" in data.frame.columns else 0
        n_times = int(data.frame["time"].nunique()) if "time" in data.frame.columns else 0
        if n_geos * n_times > 500:
            logger.warning(
                "Skipping full geo Meridian fit (%s geos × %s times). "
                "Use --dataset national (runtime default) or --dataset geo-agg.",
                n_geos,
                n_times,
            )
            return None

    try:
        input_data, channels = _build_meridian_input(data)
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M)
        )
        # Same frozen holdout as TabFM: KPI on holdout weeks excluded from training.
        model_spec = spec.ModelSpec(prior=prior, holdout_id=np.asarray(data.holdout_id))
        mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
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

        expected = ana.expected_outcome(aggregate_times=False, aggregate_geos=True, use_kpi=True)
        ts = _posterior_mean(expected)
        y_true, y_pred = _holdout_predictions_from_expected(data, ts)
        metrics = regression_metrics(y_true, y_pred)

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

        remaining_asymmetry = (
            "Shared holdout_id excludes holdout KPI from Meridian training (fair OOS KPI). "
            "Remaining Meridian design asymmetry: holdout media still enters Adstock for "
            "subsequent weeks (upstream Meridian behavior). TabFM never sees holdout rows "
            "as ICL context. MCMC here is directional PoC (tiny chains), not decision-grade."
        )

        return ModelResult(
            name="Meridian",
            mode="meridian",
            y_pred_test=np.asarray(y_pred, dtype=float),
            metrics=metrics,
            # Meridian product surface — shown in deliverable tables, not as TabFM peer contrib
            contribution_pred={
                str(k): float(v) for k, v in (deliverables.get("channel_contribution") or {}).items()
            },
            contribution_metrics={},
            extras={
                "n_keep": n_keep,
                "n_chains": 1,
                "holdout_times": data.holdout_times,
                "eval": "oos_holdout_id",
                "note": remaining_asymmetry,
                "mcmc_quality": "directional_poc_not_decision_grade",
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
