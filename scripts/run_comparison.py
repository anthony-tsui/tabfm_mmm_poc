#!/usr/bin/env python3
"""CLI: fair OOS TabFM vs Meridian comparison on official Meridian simulated data."""

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
            f"Which frozen official (or legacy) CSV to load. "
            f"PREFERRED Getting Started file: '{PREFERRED_OFFICIAL}' "
            f"(data/official/geo_all_channels.csv). "
            f"RUNTIME DEFAULT on CPU: '{DEFAULT_RUNTIME_DATASET}' "
            f"(data/official/national_all_channels.csv) because full geo MCMC is usually too heavy."
        ),
    )
    p.add_argument("--data", type=Path, default=None, help="Explicit CSV path (overrides --dataset)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip TabFM weights / Meridian MCMC; mock + Meridian-style proxy",
    )
    p.add_argument(
        "--no-meridian",
        action="store_true",
        help="Force Meridian-style proxy even if google-meridian is installed",
    )
    p.add_argument(
        "--ablation-proxy",
        action="store_true",
        help=(
            "Opt-in leave-one-channel ablation for TabFM. "
            "NOT Meridian-equivalent / not causal; excluded from headline metrics table."
        ),
    )
    p.add_argument("--train-frac", type=float, default=2 / 3)
    p.add_argument(
        "--max-context-rows",
        type=int,
        default=100,
        help="TabFM ICL context cap (Meridian still uses full non-holdout KPI weeks)",
    )
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
        ablation_proxy=args.ablation_proxy,
    )
    print_table(table)
    print_deliverables_matrix()
    print("Takeaway:", payload["practitioner_takeaway"])
    print("Disclaimer:", payload["disclaimer"])
    print(
        f"Dataset: {payload['data']['dataset']} "
        f"(preferred official={payload['data']['preferred_official']}, "
        f"runtime default={payload['data']['runtime_default']}) | "
        f"Modes: TabFM={payload['models']['tabfm']['mode']}, "
        f"Meridian={payload['models']['meridian']['mode']} | "
        f"eval={payload['fair_eval']['scoring']}"
    )
    mer = payload.get("deliverable_tables", {}).get("meridian_only_product_surface", {})
    if mer.get("roi_by_channel"):
        print("Meridian-only ROI by channel (directional PoC MCMC):", mer["roi_by_channel"])
    if mer.get("channel_contribution"):
        print(
            "Meridian-only incremental contribution (not compared to TabFM ablation):",
            mer["channel_contribution"],
        )
    if args.ablation_proxy:
        print(
            "WARNING: --ablation-proxy enabled — NOT Meridian-equivalent / not causal; "
            "see deliverable_tables.tabfm.ablation_proxy_NOT_meridian_equivalent"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
