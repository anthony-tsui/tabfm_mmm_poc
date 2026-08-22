#!/usr/bin/env python3
"""Deterministically generate Meridian-shaped synthetic MMM weekly data.

Scenario (synthetic only): Hong Kong DTC skincare, 12 weeks, six paid channels.
Planted contribution_* columns are simulation ground truth for recovery metrics —
not estimates of real campaign performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 12
START = "2024-01-01"

# Channel display names → safe column suffixes
CHANNELS = [
    ("google_sem", "Google SEM", 2.40),
    ("bing_sem", "Bing SEM", 1.80),
    ("meta", "Meta", 1.60),
    ("youtube", "YouTube", 1.40),
    ("dv360", "DV360", 1.20),
    ("ooh", "OOH", 0.90),
]

# Geometric adstock retention and Hill saturation params used in the DGP
ADSTOCK = {
    "google_sem": 0.30,
    "bing_sem": 0.25,
    "meta": 0.45,
    "youtube": 0.55,
    "dv360": 0.50,
    "ooh": 0.20,
}
HILL_EC50 = {
    "google_sem": 1.0,
    "bing_sem": 1.0,
    "meta": 1.2,
    "youtube": 1.1,
    "dv360": 1.1,
    "ooh": 0.9,
}
HILL_SLOPE = 1.5


def geometric_adstock(x: np.ndarray, retention: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + (retention * out[t - 1] if t else 0.0)
    return out


def hill(x: np.ndarray, ec50: float, slope: float = HILL_SLOPE) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return (x**slope) / (x**slope + ec50**slope + 1e-12)


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=N_WEEKS, freq="W-MON")

    # Base weekly spend (HKD) with mild seasonality + noise
    base_spend = {
        "google_sem": 85_000,
        "bing_sem": 28_000,
        "meta": 72_000,
        "youtube": 55_000,
        "dv360": 40_000,
        "ooh": 95_000,
    }
    cpm = {  # synthetic cost per thousand impressions
        "google_sem": 45.0,
        "bing_sem": 35.0,
        "meta": 28.0,
        "youtube": 55.0,
        "dv360": 48.0,
        "ooh": 120.0,
    }

    season = 1.0 + 0.08 * np.sin(np.linspace(0, 2 * np.pi, N_WEEKS))
    promo = (rng.random(N_WEEKS) < 0.25).astype(float)
    competitor = rng.normal(0.0, 1.0, N_WEEKS)
    sentiment = rng.normal(0.0, 1.0, N_WEEKS)

    rows: dict[str, np.ndarray] = {
        "time": dates.strftime("%Y-%m-%d").to_numpy(),
        "promo_control": promo,
        "competitor_activity_score_control": competitor,
        "sentiment_score_control": sentiment,
    }

    baseline = 420_000 + 8_000 * season + 35_000 * promo - 12_000 * competitor
    baseline = baseline + rng.normal(0.0, 5_000, N_WEEKS)

    incremental = np.zeros(N_WEEKS)
    for key, _label, target_roi in CHANNELS:
        spend = base_spend[key] * season * (1.0 + rng.normal(0.0, 0.12, N_WEEKS))
        spend = np.clip(spend, base_spend[key] * 0.5, None)
        impressions = (spend / cpm[key]) * 1000.0 * (1.0 + rng.normal(0.0, 0.05, N_WEEKS))
        impressions = np.clip(impressions, 0.0, None)

        # Media transform on normalized spend
        spend_norm = spend / (np.median(spend) + 1e-9)
        media = hill(geometric_adstock(spend_norm, ADSTOCK[key]), HILL_EC50[key])

        # Scale so realized ROI is near the target (contribution / spend)
        # contribution_t ≈ scale * media_t; E[contrib]/E[spend] ≈ target_roi
        scale = (target_roi * spend.mean()) / (media.mean() + 1e-9)
        contribution = scale * media + rng.normal(0.0, scale * 0.02, N_WEEKS)
        contribution = np.clip(contribution, 0.0, None)

        rows[f"{key}_spend"] = np.round(spend, 2)
        rows[f"{key}_impression"] = np.round(impressions, 0)
        rows[f"contribution_{key}"] = np.round(contribution, 2)
        incremental += contribution

        # Optional RF for video / programmatic (Meridian-compatible style)
        if key in ("youtube", "dv360"):
            reach = np.clip(impressions * rng.uniform(0.35, 0.55, N_WEEKS), 1.0, None)
            frequency = impressions / reach
            rows[f"{key}_reach"] = np.round(reach, 0)
            rows[f"{key}_frequency"] = np.round(frequency, 3)

    revenue = baseline + incremental + rng.normal(0.0, 3_000, N_WEEKS)
    rows["baseline_revenue"] = np.round(baseline, 2)
    rows["incremental_revenue"] = np.round(incremental, 2)
    rows["revenue"] = np.round(revenue, 2)
    # Meridian national sample also uses conversions + revenue_per_conversion;
    # treat revenue as the KPI directly with unit revenue.
    rows["conversions"] = np.round(revenue, 2)
    rows["revenue_per_conversion"] = np.ones(N_WEEKS)

    col_order = [
        "time",
        "conversions",
        "revenue",
        "revenue_per_conversion",
        "baseline_revenue",
        "incremental_revenue",
        "promo_control",
        "competitor_activity_score_control",
        "sentiment_score_control",
    ]
    for key, _, _ in CHANNELS:
        col_order.extend([f"{key}_spend", f"{key}_impression", f"contribution_{key}"])
        if key in ("youtube", "dv360"):
            col_order.extend([f"{key}_reach", f"{key}_frequency"])

    return pd.DataFrame(rows)[col_order]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "hk_skincare_mmm_dummy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} weeks, seed={SEED})")


if __name__ == "__main__":
    main()
