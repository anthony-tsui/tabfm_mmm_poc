#!/usr/bin/env python3
"""ONE planner story on official Meridian simulated data — visual example PNGs.

Story:
  Meridian: contribution + ROI → “cut Channel A X% / scale Channel B” annual mix.
  TabFM: holdout KPI forecast only → “stops here” (no budget/strategy).

No SHAP / ablation / feature importance as contribution.
No R²-led framing. Simulated/demo data only; tiny MCMC = directional.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmm_compare.data import load_mmm_dataset
from mmm_compare.meridian_runner import (
    _build_meridian_input,
    _posterior_mean,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("planner_story")

OUT = ROOT / "examples"
DISCLAIMER = "Official Meridian simulated/demo data — not real campaign performance"
MCMC_NOTE = "Directional PoC MCMC — not decision-grade"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _fit_meridian(data, n_keep: int = 40):
    from meridian import constants
    from meridian.analysis import analyzer
    from meridian.model import model, prior_distribution, spec
    import tensorflow_probability as tfp

    input_data, channels = _build_meridian_input(data)
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M)
    )
    model_spec = spec.ModelSpec(prior=prior, holdout_id=np.asarray(data.holdout_id))
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    mmm.sample_prior(40)
    mmm.sample_posterior(n_chains=1, n_adapt=60, n_burnin=30, n_keep=n_keep, seed=42)
    ana = analyzer.Analyzer(mmm)

    effects = _posterior_mean(
        ana.incremental_outcome(aggregate_times=True, aggregate_geos=True, use_kpi=True)
    )
    roi = _posterior_mean(ana.roi(aggregate_geos=True, use_kpi=True))
    contrib = {ch: float(max(v, 0.0)) for ch, v in zip(channels, effects[: len(channels)])}
    roi_map = {ch: float(v) for ch, v in zip(channels, roi[: len(channels)])}
    return channels, contrib, roi_map


def _build_story(data, contrib: dict[str, float], roi: dict[str, float], cut_pct: float = 0.15):
    channels = list(roi.keys())
    spend = {ch: float(data.frame[f"{ch}_spend"].sum()) for ch in channels}
    cut_ch = min(channels, key=lambda c: roi[c])
    scale_ch = max(channels, key=lambda c: roi[c])
    move = cut_pct * spend[cut_ch]
    spend_after = dict(spend)
    spend_after[cut_ch] -= move
    spend_after[scale_ch] += move
    # Illustrative incremental using Meridian ROI × spend (scenario, not full optimizer)
    before_inc = sum(roi[c] * spend[c] for c in channels)
    after_inc = sum(roi[c] * spend_after[c] for c in channels)
    story = {
        "cut_channel": cut_ch,
        "scale_channel": scale_ch,
        "cut_pct": cut_pct,
        "move_spend": move,
        "spend_before": spend,
        "spend_after": spend_after,
        "roi": roi,
        "contrib": contrib,
        "before_incremental": before_inc,
        "after_incremental": after_inc,
        "delta_incremental": after_inc - before_inc,
        "caption_meridian": (
            f"Cut {cut_ch} {cut_pct:.0%} / scale {scale_ch} — "
            f"lowest ROI (~{roi[cut_ch]:.0f}) vs highest (~{roi[scale_ch]:.0f}). "
            f"That’s an annual-mix call."
        ),
        "caption_tabfm": "KPI forecast only — planner stops here.",
    }
    return story


def plot_meridian_story(story: dict) -> None:
    """Single visual story image: contribution + ROI → cut A / scale B with numbers."""
    channels = list(story["roi"].keys())
    cut_ch, scale_ch = story["cut_channel"], story["scale_channel"]
    contrib = np.array([story["contrib"][c] for c in channels], dtype=float)
    roi_vals = np.array([story["roi"][c] for c in channels], dtype=float)
    before = np.array([story["spend_before"][c] for c in channels], dtype=float)
    after = np.array([story["spend_after"][c] for c in channels], dtype=float)

    fig = plt.figure(figsize=(8.2, 13.0))
    # Charts + legend strip + caption — caption never shares space with x-ticks
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1.0, 1.0, 1.15, 0.32, 0.7], hspace=0.45)

    fig.suptitle(
        "Planner story — Meridian on simulated data\n"
        "Annual mix: what to cut vs scale",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax0 = fig.add_subplot(gs[0])
    colors = ["#cf222e" if c == cut_ch else "#2da44e" if c == scale_ch else "#6e7781" for c in channels]
    ax0.barh(channels[::-1], (contrib / 1e9)[::-1], color=colors[::-1])
    ax0.set_xlabel("Channel contribution (billions, simulated)")
    ax0.set_title("1. What drove the outcome?  (Meridian incremental contribution)", loc="left", pad=8)

    ax1 = fig.add_subplot(gs[1])
    ax1.bar(channels, roi_vals, color=colors)
    ax1.set_ylabel("ROI (simulated)")
    ax1.set_title("2. Which channels are efficient?  (Meridian ROI)", loc="left", pad=8)
    ax1.tick_params(axis="x", rotation=20)

    ax2 = fig.add_subplot(gs[2])
    x = np.arange(len(channels))
    w = 0.36
    bars_before = ax2.bar(x - w / 2, before / 1e6, width=w, label="Before", color="#8c959f")
    bars_after = ax2.bar(x + w / 2, after / 1e6, width=w, label="After scenario", color="#1f6feb")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, rotation=20)
    ax2.set_ylabel("Spend (millions, simulated)")
    ax2.set_title(
        f"3. Sample annual mix rec: cut {cut_ch} {story['cut_pct']:.0%} → scale {scale_ch}",
        loc="left",
        pad=8,
    )

    ax_leg = fig.add_subplot(gs[3])
    ax_leg.axis("off")
    ax_leg.legend(
        [bars_before, bars_after],
        ["Before", "After scenario"],
        loc="center",
        ncol=2,
        fontsize=10,
        frameon=False,
    )

    ax_cap = fig.add_subplot(gs[4])
    ax_cap.axis("off")
    move_m = story["move_spend"] / 1e6
    delta_b = story["delta_incremental"] / 1e9
    caption = (
        f"Cut {cut_ch} {story['cut_pct']:.0%} / scale {scale_ch} — "
        f"lowest ROI (~{story['roi'][cut_ch]:.0f}) vs highest (~{story['roi'][scale_ch]:.0f}). "
        f"That’s an annual-mix call.\n"
        f"(Moved ${move_m:.1f}M simulated spend; illustrative incremental "
        f"{'+' if delta_b >= 0 else ''}{delta_b:.2f}B via ROI×spend scenario — not full optimizer.)\n"
        f"{DISCLAIMER}. {MCMC_NOTE}."
    )
    ax_cap.text(0.0, 0.9, caption, fontsize=10, color="#24292f", va="top", ha="left", wrap=True)
    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.12, right=0.96)
    _save(fig, "01_meridian_annual_mix_story.png")


def plot_tabfm_story(data, y_pred: np.ndarray) -> None:
    """KPI forecast + explicit stops-here panel — clear margins, no overlapping labels."""
    y_true = data.y_test.to_numpy(dtype=float)
    n = min(len(y_true), len(y_pred), 52)

    fig = plt.figure(figsize=(8.2, 10.0))
    # Chart | legend strip | stop box | caption — each owns its row (no shared space)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[3.0, 0.38, 2.4, 0.55], hspace=0.28)

    fig.suptitle(
        "Same planner question — TabFM on the same simulated data",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    ax0 = fig.add_subplot(gs[0])
    (ln0,) = ax0.plot(
        range(n), y_true[:n] / 1e6, marker="o", ms=3.5, label="Actual holdout KPI", color="#24292f"
    )
    (ln1,) = ax0.plot(
        range(n), y_pred[:n] / 1e6, marker="s", ms=3.5, label="TabFM predicted KPI", color="#bf3989"
    )
    ax0.set_xlabel("Holdout week")
    ax0.set_ylabel("KPI (millions, simulated)")
    ax0.set_title("Holdout KPI: actual vs TabFM", loc="left", pad=10)
    ax0.set_ylim(
        min(y_true[:n].min(), y_pred[:n].min()) / 1e6 * 0.98,
        max(y_true[:n].max(), y_pred[:n].max()) / 1e6 * 1.04,
    )

    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")
    ax_leg.legend(
        [ln0, ln1],
        ["Actual holdout KPI", "TabFM predicted KPI"],
        loc="center",
        ncol=2,
        fontsize=10,
        frameon=False,
    )

    ax1 = fig.add_subplot(gs[2])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.add_patch(
        plt.Rectangle(
            (0.35, 0.4), 9.3, 9.2, fill=True, color="#fff5f5", ec="#cf222e", lw=2.5, clip_on=False
        )
    )
    ax1.text(
        5, 8.2, "Planner stops here", ha="center", va="center", fontsize=17, fontweight="bold", color="#cf222e"
    )
    ax1.text(
        5,
        5.5,
        "No contribution   ·   No ROI\nNo cut / scale   ·   No annual mix plan",
        ha="center",
        va="center",
        fontsize=12,
        color="#24292f",
        linespacing=1.75,
    )
    ax1.text(
        5,
        2.0,
        "Needs Meridian (or another MMM) for annual\nbudget & strategy recommendations.",
        ha="center",
        va="center",
        fontsize=11,
        color="#57606a",
        linespacing=1.45,
    )

    ax_cap = fig.add_subplot(gs[3])
    ax_cap.axis("off")
    ax_cap.text(
        0.0,
        0.75,
        f"KPI forecast only — planner stops here.\n{DISCLAIMER}.",
        fontsize=11,
        color="#24292f",
        va="top",
        ha="left",
    )
    fig.subplots_adjust(top=0.93, bottom=0.04, left=0.12, right=0.96)
    _save(fig, "02_tabfm_kpi_stops_here.png")


def plot_story_strip() -> None:
    """Composite strip: stack Meridian + TabFM panels (source images already captioned)."""
    p1, p2 = OUT / "01_meridian_annual_mix_story.png", OUT / "02_tabfm_kpi_stops_here.png"
    if not (p1.exists() and p2.exists()):
        return
    # Avoid stacking duplicate titles/captions on top of the source PNGs —
    # those already include story text with clear margins.
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 16.0))
    for ax, path in zip(axes, (p1, p2)):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    fig.suptitle("One planner story — Meridian vs TabFM", fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.97, bottom=0.01, left=0.02, right=0.98, hspace=0.06)
    _save(fig, "00_one_planner_story.png")


def main() -> int:
    _style()
    data = load_mmm_dataset(dataset="national", max_context_rows=100)
    logger.info("Frozen slice: %s | channels=%s", data.source_path, data.channel_keys)

    saved = OUT / "story_numbers.json"
    metrics_path = ROOT / "results" / "metrics.json"
    use_saved = "--from-saved" in sys.argv and saved.exists()

    if use_saved and metrics_path.exists():
        y_pred = np.asarray(
            json.loads(metrics_path.read_text())["models"]["tabfm"]["y_pred_test"],
            dtype=float,
        )
        logger.info("TabFM preds from %s (no TabFM reload)", metrics_path)
    else:
        from mmm_compare.tabfm_runner import run_tabfm

        tabfm = run_tabfm(data, dry_run=False)
        y_pred = tabfm.y_pred_test
    plot_tabfm_story(data, y_pred)

    if use_saved:
        prev = json.loads(saved.read_text())
        contrib = prev["story"]["contrib"]
        roi = prev["story"]["roi"]
        story = _build_story(data, contrib, roi, cut_pct=float(prev["story"].get("cut_pct", 0.15)))
        logger.info("Rebuilt story from %s (no Meridian refit)", saved)
    else:
        try:
            _channels, contrib, roi = _fit_meridian(data)
            story = _build_story(data, contrib, roi)
        except Exception as exc:
            logger.warning(
                "Meridian fit failed (%s) — cannot invent TabFM strategy; abort Meridian story",
                exc,
            )
            raise

    plot_meridian_story(story)
    plot_story_strip()

    # Numbers for the markdown story (no metric bake-off)
    payload = {
        "disclaimer": DISCLAIMER,
        "mcmc_note": MCMC_NOTE,
        "dataset": data.source_path,
        "channels": data.channel_keys,
        "story": {
            "cut_channel": story["cut_channel"],
            "scale_channel": story["scale_channel"],
            "cut_pct": story["cut_pct"],
            "move_spend": story["move_spend"],
            "roi_cut": story["roi"][story["cut_channel"]],
            "roi_scale": story["roi"][story["scale_channel"]],
            "delta_incremental": story["delta_incremental"],
            "caption_meridian": story["caption_meridian"],
            "caption_tabfm": story["caption_tabfm"],
            "roi": story["roi"],
            "contrib": story["contrib"],
            "spend_before": story["spend_before"],
            "spend_after": story["spend_after"],
        },
    }
    (OUT / "story_numbers.json").write_text(json.dumps(payload, indent=2))
    logger.info("Story: cut %s / scale %s", story["cut_channel"], story["scale_channel"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
