#!/usr/bin/env python3
"""CLI: compare TabFM vs Meridian on official Meridian simulated data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmm_compare.compare import print_deliverables_matrix, print_table, run_comparison
from mmm_compare.data import DATASET_FILES, DEFAULT_RUNTIME_DATASET, PREFERRED_OFFICIAL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=sorted(DATASET_FILES),
        default=DEFAULT_RUNTIME_DATASET,
        help=(
            f"Official Meridian simulated dataset key (default: {DEFAULT_RUNTIME_DATASET}). "
            f"Preferred Getting Started file is '{PREFERRED_OFFICIAL}' but full geo MCMC "
            "is usually too heavy on CPU — see README."
        ),
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional explicit CSV path (overrides --dataset)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip TabFM weight download / Meridian MCMC; use mock + Meridian-style proxy",
    )
    p.add_argument(
        "--no-meridian",
        action="store_true",
        help="Force Meridian-style proxy even if google-meridian is installed",
    )
    p.add_argument("--train-frac", type=float, default=2 / 3)
    p.add_argument("--max-context-rows", type=int, default=100, help="TabFM ICL context cap")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    table, _results, payload = run_comparison(
        args.data,
        dataset=None if args.data else args.dataset,
        dry_run=args.dry_run,
        prefer_real_meridian=not args.no_meridian,
        train_frac=args.train_frac,
        results_dir=args.results_dir,
        max_context_rows=args.max_context_rows,
    )
    print_table(table)
    print_deliverables_matrix()
    print("Disclaimer:", payload["disclaimer"])
    print(
        f"Dataset: {payload['data']['dataset']} | "
        f"Modes: TabFM={payload['models']['tabfm']['mode']}, "
        f"Meridian={payload['models']['meridian']['mode']}"
    )
    mer_del = payload.get("deliverable_tables", {}).get("meridian", {})
    if mer_del.get("roi_by_channel"):
        print("Meridian ROI by channel:", mer_del["roi_by_channel"])
    if mer_del.get("channel_contribution"):
        print("Meridian channel contribution:", mer_del["channel_contribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
