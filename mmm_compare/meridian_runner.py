"""Meridian baseline with Meridian-style lightweight proxy fallback."""

from __future__ import annotations

import logging

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
    """Lightweight media-mix style proxy (adstock + Hill + Ridge).

    Labeled clearly as Meridian-style baseline — not full Google Meridian MCMC.
    """
    X_all, media_cols = _media_design_matrix(data.frame, data.channel_keys)
    X_train = X_all.loc[data.train_idx]
    X_test = X_all.loc[data.test_idx]
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_train, data.y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(data.y_test, pred)

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


def _posterior_mean_time_series(tensor) -> np.ndarray:
    """Reduce Meridian (chain, draw, time[, ...]) tensors to a time vector."""
    arr = np.asarray(tensor, dtype=float)
    # Common shapes: (chains, draws, time) or (chains, draws, time, channel)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    return arr


def _try_meridian(data: MMMDataset, *, n_keep: int = 50) -> ModelResult | None:
    """Best-effort minimal Meridian fit on national weekly data."""
    try:
        from meridian import constants
        from meridian.analysis import analyzer
        from meridian.data import data_frame_input_data_builder
        from meridian.model import model, prior_distribution, spec
        import tensorflow_probability as tfp
    except Exception as exc:
        logger.info("google-meridian not available (%s)", exc)
        return None

    df = data.frame.copy()
    if "revenue_per_conversion" not in df.columns:
        df["revenue_per_conversion"] = 1.0

    channels: list[str] = []
    media_cols: list[str] = []
    spend_cols: list[str] = []
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
    kpi_col = data.kpi_col if data.kpi_col in df.columns else "conversions"

    try:
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
            n_chains=1,
            n_adapt=80,
            n_burnin=40,
            n_keep=n_keep,
            seed=42,
        )

        ana = analyzer.Analyzer(mmm)
        # Shape (chains, draws, time); mean over posterior → weekly expected KPI.
        expected = ana.expected_outcome(aggregate_times=False, aggregate_geos=True)
        time_series = _posterior_mean_time_series(expected)
        if len(time_series) != len(df):
            logger.warning(
                "Meridian prediction length mismatch (%s vs %s)",
                len(time_series),
                len(df),
            )
            return None

        # Note: Meridian is fit on the full series here (national PoC). Holdout
        # KPI metrics are therefore in-sample expected outcome on later weeks —
        # not a pure OOS forecast. Documented in README.
        pred = time_series[np.asarray(data.test_idx)]
        metrics = regression_metrics(data.y_test, pred)

        contrib: dict[str, float] = {}
        try:
            # Shape (chains, draws, channel) with aggregate_times=True
            effects = ana.incremental_outcome(aggregate_times=True, aggregate_geos=True)
            e = np.asarray(effects, dtype=float)
            while e.ndim > 1:
                e = e.mean(axis=0)
            if len(e) == len(channels):
                # Convert period totals to mean-per-holdout-week scale for share metrics
                n_test = max(len(data.test_idx), 1)
                n_all = max(len(df), 1)
                # Approximate holdout-period contribution as proportional share of total
                scale = n_test / n_all
                contrib = {ch: float(max(v, 0.0) * scale) for ch, v in zip(channels, e)}
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
            extras={
                "n_keep": n_keep,
                "n_chains": 1,
                "note": (
                    "Minimal Meridian MCMC on full national series; holdout KPI "
                    "uses in-sample expected_outcome on later weeks (not pure OOS)."
                ),
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
