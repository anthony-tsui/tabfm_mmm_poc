"""Meridian baseline with Meridian-style lightweight proxy fallback."""

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
    """Build adstock+Hill transformed spend features + controls."""
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
        if c.endswith("_control"):
            parts[c] = df[c].to_numpy(dtype=float)
    X = pd.DataFrame(parts, index=df.index)
    return X, media_cols


def run_meridian_style(data: MMMDataset) -> ModelResult:
    """Lightweight Bayesian/ridge media-mix style proxy (adstock + Hill + Ridge).

    Labeled clearly as Meridian-style baseline — not full Google Meridian MCMC.
    """
    X_all, media_cols = _media_design_matrix(data.frame, data.channel_keys)
    X_train = X_all.loc[data.train_idx]
    X_test = X_all.loc[data.test_idx]
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_train, data.y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(data.y_test, pred)

    # Channel contributions ≈ coef * transformed media on holdout
    contrib: dict[str, float] = {}
    coef_map = dict(zip(X_all.columns, model.coef_))
    for key in data.channel_keys:
        col = f"{key}_media_transform"
        if col in coef_map:
            contrib[key] = float(np.mean(np.maximum(coef_map[col] * X_test[col].to_numpy(), 0.0)))

    true_c = _true_holdout_contributions(data)
    c_metrics = contribution_recovery_metrics(true_c, contrib) if true_c else {}
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
                "Not full Google Meridian Bayesian MCMC."
            ),
            "media_cols": media_cols,
        },
    )


def _try_meridian(data: MMMDataset, *, n_keep: int = 100) -> ModelResult | None:
    """Best-effort minimal Meridian fit on national weekly data."""
    try:
        from meridian.data import data_frame_input_data_builder
        from meridian.model import model, prior_distribution, spec
        import tensorflow_probability as tfp
        from meridian import constants
    except Exception as exc:
        logger.info("google-meridian not available (%s)", exc)
        return None

    df = data.frame.copy()
    # Meridian expects conversions-style KPI; map our revenue KPI if needed.
    if "conversions" not in df.columns:
        df["conversions"] = df[data.kpi_col]
    if "revenue_per_conversion" not in df.columns:
        df["revenue_per_conversion"] = 1.0

    channels = []
    media_cols = []
    spend_cols = []
    for key in data.channel_keys:
        imp = f"{key}_impression"
        sp = f"{key}_spend"
        if imp in df.columns and sp in df.columns:
            channels.append(key)
            media_cols.append(imp)
            spend_cols.append(sp)
    if not channels:
        logger.warning("No impression/spend pairs for Meridian; skipping")
        return None

    control_cols = [c for c in df.columns if c.endswith("_control")]
    try:
        builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
            kpi_type="revenue" if data.kpi_col == "revenue" else "non_revenue",
            default_kpi_column="conversions" if data.kpi_col != "revenue" else data.kpi_col,
            default_revenue_per_kpi_column="revenue_per_conversion",
        )
        # Prefer revenue KPI when present
        kpi_col = data.kpi_col if data.kpi_col in df.columns else "conversions"
        builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
            kpi_type="revenue",
            default_kpi_column=kpi_col,
            default_revenue_per_kpi_column="revenue_per_conversion",
        )
        builder = builder.with_kpi(df).with_revenue_per_kpi(df)
        if control_cols:
            builder = builder.with_controls(df, control_cols=control_cols)
        builder = builder.with_media(
            df,
            media_cols=media_cols,
            media_spend_cols=spend_cols,
            media_channels=channels,
        )
        input_data = builder.build()

        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M)
        )
        model_spec = spec.ModelSpec(prior=prior)
        mmm = model.Meridian(input_data=input_data, model_spec=model_spec)

        # Tiny sampling for PoC CPU friendliness — not production-quality chains.
        mmm.sample_prior(50)
        mmm.sample_posterior(
            n_chains=2,
            n_adapt=100,
            n_burnin=50,
            n_keep=n_keep,
            seed=42,
        )

        # Predictive holdout via in-sample media contribution aggregation is complex;
        # use fitted expected outcome on all times and slice holdout weeks.
        # Fall back to meridian_style metrics if analyzer API differs across versions.
        from meridian.analysis import analyzer

        ana = analyzer.Analyzer(mmm)
        # expected_outcome shape depends on version; try common paths
        expected = None
        for method_name in ("expected_outcome", "expected_kpi"):
            method = getattr(ana, method_name, None)
            if callable(method):
                try:
                    expected = method()
                    break
                except TypeError:
                    try:
                        expected = method(aggregate_times=False)
                        break
                    except Exception:
                        continue
        if expected is None:
            logger.warning("Could not extract Meridian expected outcome; using style proxy metrics")
            style = run_meridian_style(data)
            style.extras["meridian_fit"] = "sampled_but_prediction_extract_failed"
            style.name = "Meridian (fit ok; proxy metrics)"
            style.mode = "meridian"
            return style

        arr = np.asarray(expected)
        # Reduce to time dimension
        while arr.ndim > 1:
            arr = arr.mean(axis=0)
        if len(arr) != len(df):
            logger.warning("Meridian prediction length mismatch (%s vs %s)", len(arr), len(df))
            return None
        pred = arr[data.test_idx]
        metrics = regression_metrics(data.y_test, pred)

        # Contribution recovery from media effects if available
        contrib: dict[str, float] = {}
        media_effect = getattr(ana, "media_effects", None) or getattr(ana, "incremental_outcome", None)
        if callable(media_effect):
            try:
                effects = media_effect()
                e = np.asarray(effects)
                # Best-effort: last dim channels
                if e.ndim >= 1 and e.shape[-1] == len(channels):
                    # average over other dims, then over train+test → use test slice if time present
                    while e.ndim > 2:
                        e = e.mean(axis=0)
                    if e.ndim == 2 and e.shape[0] == len(df):
                        e = e[data.test_idx].mean(axis=0)
                    else:
                        e = e.mean(axis=tuple(range(e.ndim - 1)))
                    contrib = {ch: float(max(v, 0.0)) for ch, v in zip(channels, e)}
            except Exception as exc:
                logger.info("Meridian contribution extract skipped: %s", exc)

        true_c = _true_holdout_contributions(data)
        c_metrics = contribution_recovery_metrics(true_c, contrib) if true_c and contrib else {}
        return ModelResult(
            name="Meridian",
            mode="meridian",
            y_pred_test=np.asarray(pred, dtype=float),
            metrics=metrics,
            contribution_pred=contrib,
            contribution_metrics=c_metrics,
            extras={"n_keep": n_keep, "n_chains": 2, "note": "Minimal Meridian MCMC PoC settings"},
        )
    except Exception as exc:
        logger.warning("Meridian fit failed (%s)", exc)
        return None


def run_meridian_baseline(
    data: MMMDataset,
    *,
    prefer_real: bool = True,
    dry_run: bool = False,
    n_keep: int = 100,
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
