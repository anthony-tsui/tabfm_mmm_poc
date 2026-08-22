#!/usr/bin/env python3
"""Curiosity overlay: Actual vs Meridian vs TabFM on shared holdout (from metrics.json).

Does not change the annual-mix success bar. Caption: similar KPI fit ≠ same model
for annual mix. Overlay is curiosity only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmm_compare.data import load_mmm_dataset

OUT = ROOT / "examples"
METRICS = ROOT / "results" / "metrics.json"
DISCLAIMER = "Official Meridian simulated/demo data — not real campaign performance"
CAPTION = "Similar KPI fit ≠ same model for annual mix. Overlay is curiosity only."


def main() -> int:
    payload = json.loads(METRICS.read_text())
    holdout_times = payload["fair_eval"]["shared_holdout_times"]
    tabfm = payload["models"]["tabfm"]
    meridian = payload["models"]["meridian"]
    y_tab = np.asarray(tabfm["y_pred_test"], dtype=float)
    y_mer = np.asarray(meridian["y_pred_test"], dtype=float)

    data = load_mmm_dataset(dataset="national", max_context_rows=100)
    times = [str(t) for t in data.frame.loc[data.test_idx, "time"].tolist()]
    y_act = data.y_test.to_numpy(dtype=float)
    if times != holdout_times:
        raise SystemExit(
            f"Holdout mismatch vs metrics.json ({times[0]}..{times[-1]} vs "
            f"{holdout_times[0]}..{holdout_times[-1]})"
        )
    if not (len(y_act) == len(y_tab) == len(y_mer) == 52):
        raise SystemExit(f"Length mismatch act={len(y_act)} tab={len(y_tab)} mer={len(y_mer)}")

    dates = pd.to_datetime(times)
    tm, mm = tabfm["metrics"], meridian["metrics"]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
            "figure.dpi": 140,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    ax.plot(dates, y_act / 1e6, color="#24292f", lw=2.0, marker="o", ms=3.5, label="Actual KPI")
    ax.plot(dates, y_mer / 1e6, color="#1f6feb", lw=2.0, marker="^", ms=3.5, label="Meridian holdout forecast")
    ax.plot(dates, y_tab / 1e6, color="#bf3989", lw=2.0, marker="s", ms=3.5, label="TabFM holdout forecast")
    ax.set_ylabel("KPI (millions, simulated conversions)")
    ax.set_xlabel("Shared holdout weeks")
    ax.set_title(
        f"Curiosity overlay — same holdout\n{holdout_times[0]} → {holdout_times[-1]} (52 weeks)",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=25, ha="right")
    ax.legend(loc="upper left", fontsize=10)

    # Multi-metric card (not R²-only)
    card = (
        "Fair OOS metrics (from results/metrics.json)\n"
        f"{'':8} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'R²':>7}\n"
        f"{'Meridian':8} {mm['rmse']/1e6:10.2f}M {mm['mae']/1e6:10.2f}M {mm['mape_pct']:8.2f} {mm['r2']:7.3f}\n"
        f"{'TabFM':8} {tm['rmse']/1e6:10.2f}M {tm['mae']/1e6:10.2f}M {tm['mape_pct']:8.2f} {tm['r2']:7.3f}"
    )
    ax.text(
        0.98,
        0.02,
        card,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f8fa", edgecolor="#d0d7de"),
    )

    fig.text(
        0.02,
        0.01,
        f"{CAPTION}\n"
        f"Does not change the annual-mix bar — TabFM still has no cut/scale / ROI / contribution.\n"
        f"{DISCLAIMER}.",
        fontsize=10,
        color="#24292f",
        va="bottom",
    )
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "03_curiosity_kpi_overlay.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    print(f"Window {holdout_times[0]} → {holdout_times[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
