#!/usr/bin/env python3
"""CLI: compare TabFM vs Meridian(-style) on synthetic MMM data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmm_compare.compare import print_table, run_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "hk_skincare_mmm_dummy.csv",
        help="Path to Meridian-shaped weekly CSV",
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
    p.add_argument(
        "--train-frac",
        type=float,
        default=2 / 3,
        help="Fraction of earlier weeks used as train/context",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory for metrics.json / metrics.csv",
    )
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
        dry_run=args.dry_run,
        prefer_real_meridian=not args.no_meridian,
        train_frac=args.train_frac,
        results_dir=args.results_dir,
    )
    print_table(table)
    print("Disclaimer:", payload["disclaimer"])
    print(f"Modes: TabFM={payload['models']['tabfm']['mode']}, "
          f"Meridian={payload['models']['meridian']['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
