"""Orchestrate TabFM vs Meridian(-style) comparison and persist metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmm_compare.data import load_mmm_dataset
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
    row = {
        "model": result.name,
        "mode": result.mode,
        **{f"kpi_{k}": v for k, v in result.metrics.items()},
        **{f"recovery_{k}": v for k, v in result.contribution_metrics.items()},
    }
    return row


def run_comparison(
    data_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    prefer_real_meridian: bool = True,
    train_frac: float = 2 / 3,
    results_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, ModelResult], dict[str, Any]]:
    data = load_mmm_dataset(data_path, train_frac=train_frac)
    logger.info(
        "Loaded %s rows | KPI=%s | train=%s test=%s | features=%s",
        len(data.frame),
        data.kpi_col,
        len(data.train_idx),
        len(data.test_idx),
        len(data.feature_cols),
    )

    tabfm = run_tabfm(data, dry_run=dry_run)
    meridian = run_meridian_baseline(data, prefer_real=prefer_real_meridian, dry_run=dry_run)

    results = {"tabfm": tabfm, "meridian": meridian}
    table = metrics_table([result_to_row(tabfm), result_to_row(meridian)])

    payload = {
        "disclaimer": (
            "All values are synthetic/dummy. Metrics do not estimate real campaign performance."
        ),
        "data": {
            "path": str(data_path or "data/hk_skincare_mmm_dummy.csv"),
            "n_rows": len(data.frame),
            "kpi": data.kpi_col,
            "train_idx": data.train_idx,
            "test_idx": data.test_idx,
            "feature_cols": data.feature_cols,
            "channel_keys": data.channel_keys,
            "has_contribution_ground_truth": bool(data.contribution_cols),
        },
        "dry_run": dry_run,
        "models": {k: _jsonable(v) for k, v in results.items()},
        "metrics_table": table.reset_index().to_dict(orient="records"),
    }

    if results_dir is not None:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = "metrics_dry_run" if dry_run else "metrics"
        json_path = out / f"{stem}.json"
        csv_path = out / f"{stem}.csv"
        json_path.write_text(json.dumps(payload, indent=2))
        table.to_csv(csv_path)
        logger.info("Wrote %s and %s", json_path, csv_path)

    return table, results, payload


def print_table(table: pd.DataFrame) -> None:
    print("\n=== Side-by-side holdout metrics (synthetic data only) ===")
    print(table.to_string(float_format=lambda x: f"{x:,.4f}"))
    print()
