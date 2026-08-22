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
from matplotlib.gridspec import GridSpec

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

    fig = plt.figure(figsize=(8.4, 8.2))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3.2, 1.0, 0.7], hspace=0.35)

    ax = fig.add_subplot(gs[0])
    ax.plot(dates, y_act / 1e6, color="#24292f", lw=2.0, marker="o", ms=3.5, label="Actual KPI")
    ax.plot(dates, y_mer / 1e6, color="#1f6feb", lw=2.0, marker="^", ms=3.5, label="Meridian holdout forecast")
    ax.plot(dates, y_tab / 1e6, color="#bf3989", lw=2.0, marker="s", ms=3.5, label="TabFM holdout forecast")
    ax.set_ylabel("KPI (millions, simulated conversions)")
    ax.set_xlabel("Shared holdout weeks")
    ax.set_title(
        f"Curiosity overlay — same holdout\n{holdout_times[0]} → {holdout_times[-1]} (52 weeks)",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Metrics card in its own panel — never over the series
    ax_m = fig.add_subplot(gs[1])
    ax_m.axis("off")
    card = (
        "Fair OOS metrics (from results/metrics.json)\n"
        f"{'':10} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'R²':>7}\n"
        f"{'Meridian':10} {mm['rmse']/1e6:10.2f}M {mm['mae']/1e6:10.2f}M {mm['mape_pct']:8.2f} {mm['r2']:7.3f}\n"
        f"{'TabFM':10} {tm['rmse']/1e6:10.2f}M {tm['mae']/1e6:10.2f}M {tm['mape_pct']:8.2f} {tm['r2']:7.3f}"
    )
    ax_m.text(
        0.5,
        0.5,
        card,
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f6f8fa", edgecolor="#d0d7de"),
    )

    ax_c = fig.add_subplot(gs[2])
    ax_c.axis("off")
    ax_c.text(
        0.0,
        0.85,
        f"{CAPTION}\n"
        f"Does not change the annual-mix bar — TabFM still has no cut/scale / ROI / contribution.\n"
        f"{DISCLAIMER}.",
        fontsize=10,
        color="#24292f",
        va="top",
        ha="left",
    )

    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.12, right=0.96)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "03_curiosity_kpi_overlay.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    print(f"Window {holdout_times[0]} → {holdout_times[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
