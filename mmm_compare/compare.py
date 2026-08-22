"""Orchestrate TabFM vs Meridian: fair OOS predictive metrics + deliverables gap analysis."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm_compare.data import DEFAULT_RUNTIME_DATASET, PREFERRED_OFFICIAL, load_mmm_dataset
from mmm_compare.deliverables import capability_summary, deliverables_dataframe, deliverables_markdown
from mmm_compare.meridian_runner import run_meridian_baseline
from mmm_compare.metrics import metrics_table
from mmm_compare.tabfm_runner import ModelResult, run_tabfm

logger = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def result_to_row(result: ModelResult) -> dict[str, Any]:
    """Headline metrics table: predictive KPI only (no contribution peer columns)."""
    return {
        "model": result.name,
        "mode": result.mode,
        **{f"kpi_{k}": v for k, v in result.metrics.items()},
    }


def _deliverable_tables(results: dict[str, ModelResult]) -> dict[str, Any]:
    mer = results["meridian"].extras.get("deliverables") or {}
    tab = results["tabfm"].extras.get("deliverables") or {}
    return {
        "meridian_only_product_surface": {
            "mode": results["meridian"].mode,
            "mcmc_quality": results["meridian"].extras.get("mcmc_quality")
            or mer.get("mcmc_quality"),
            "predictive_accuracy": mer.get("predictive_accuracy"),
            "channel_contribution": mer.get("channel_contribution"),
            "roi_by_channel": mer.get("roi_by_channel"),
            "summary_metrics_preview": (
                (mer.get("summary_metrics") or [])[:15]
                if isinstance(mer.get("summary_metrics"), list)
                else mer.get("summary_metrics")
            ),
            "response_curves": mer.get("response_curves"),
            "budget_optimization": mer.get("budget_optimization"),
            "geo_insights": mer.get("geo_insights"),
            "extraction_notes": mer.get("extraction_notes"),
        },
        "tabfm": {
            "mode": results["tabfm"].mode,
            "predictive_kpi_only": True,
            "honesty": tab.get("honesty"),
            "ablation_proxy_NOT_meridian_equivalent": tab.get(
                "ablation_proxy_NOT_meridian_equivalent"
            ),
        },
    }


def run_comparison(
    data_path: str | Path | None = None,
    *,
    dataset: str | None = None,
    dry_run: bool = False,
    prefer_real_meridian: bool = True,
    train_frac: float = 2 / 3,
    results_dir: str | Path | None = None,
    max_context_rows: int = 100,
    ablation_proxy: bool = False,
) -> tuple[pd.DataFrame, dict[str, ModelResult], dict[str, Any]]:
    data = load_mmm_dataset(
        data_path,
        dataset=dataset,
        train_frac=train_frac,
        max_context_rows=max_context_rows,
    )
    logger.info(
        "Frozen input dataset=%s path=%s rows=%s KPI=%s "
        "train=%s holdout=%s tabfm_context=%s features=%s | %s",
        data.dataset_name,
        data.source_path,
        len(data.frame),
        data.kpi_col,
        len(data.train_idx),
        len(data.test_idx),
        len(data.tabfm_context_idx),
        len(data.feature_cols),
        data.notes,
    )

    tabfm = run_tabfm(data, dry_run=dry_run, ablation_proxy=ablation_proxy)
    meridian = run_meridian_baseline(data, prefer_real=prefer_real_meridian, dry_run=dry_run)

    results = {"tabfm": tabfm, "meridian": meridian}
    table = metrics_table([result_to_row(tabfm), result_to_row(meridian)])
    matrix = deliverables_dataframe()
    caps = capability_summary()
    deliverable_tables = _deliverable_tables(results)

    payload = {
        "disclaimer": (
            "Official Meridian simulated/demo data only. Not real campaign performance. "
            "Predictive fit ≠ media decisioning."
        ),
        "practitioner_takeaway": caps["headline"],
        "fair_eval": {
            "frozen_input": data.source_path,
            "shared_holdout_times": data.holdout_times,
            "holdout_id_shape": list(np.asarray(data.holdout_id).shape),
            "tabfm_context_rows": len(data.tabfm_context_idx),
            "meridian_train_kpi_rows": len(data.train_idx),
            "scoring": (
                "Both models scored on the same later-week holdout KPI. "
                "Meridian uses ModelSpec.holdout_id (KPI excluded from training). "
                "TabFM uses train rows as ICL context only."
            ),
            "remaining_asymmetry": results["meridian"].extras.get("note"),
        },
        "data": {
            "dataset": data.dataset_name,
            "preferred_official": PREFERRED_OFFICIAL,
            "runtime_default": DEFAULT_RUNTIME_DATASET,
            "path": data.source_path,
            "n_rows": len(data.frame),
            "is_geo": data.is_geo,
            "kpi": data.kpi_col,
            "n_train": len(data.train_idx),
            "n_test": len(data.test_idx),
            "feature_cols": data.feature_cols,
            "channel_keys": data.channel_keys,
            "notes": data.notes,
        },
        "dry_run": dry_run,
        "ablation_proxy": ablation_proxy,
        "models": {k: _jsonable(v) for k, v in results.items()},
        "metrics_table": table.reset_index().to_dict(orient="records"),
        "deliverables_matrix": matrix.to_dict(orient="records"),
        "deliverables_capability_summary": caps,
        "deliverable_tables": _jsonable(deliverable_tables),
    }

    if results_dir is not None:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = "metrics_dry_run" if dry_run else "metrics"
        (out / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        table.to_csv(out / f"{stem}.csv")
        matrix.to_csv(out / "deliverables_matrix.csv", index=False)
        (out / "deliverables_matrix.md").write_text(
            "# Meridian vs TabFM deliverables\n\n"
            + caps["headline"]
            + "\n\n"
            + deliverables_markdown()
            + "\n"
        )
        logger.info("Wrote results under %s", out)

    return table, results, payload


def print_table(table: pd.DataFrame) -> None:
    print("\n=== Holdout predictive KPI (same frozen table + shared holdout; simulated data) ===")
    print(table.to_string(float_format=lambda x: f"{x:,.4f}"))
    print()


def print_deliverables_matrix() -> None:
    caps = capability_summary()
    print("\n=== Deliverables gap analysis (not a bake-off win framing) ===")
    print(caps["headline"])
    print()
    print(deliverables_dataframe().to_string(index=False))
    print()
