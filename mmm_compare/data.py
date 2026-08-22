"""Load and split Meridian-shaped synthetic MMM data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "hk_skincare_mmm_dummy.csv"

CHANNEL_KEYS = [
    "google_sem",
    "bing_sem",
    "meta",
    "youtube",
    "dv360",
    "ooh",
]

KPI_CANDIDATES = ("revenue", "conversions")
CONTROL_SUFFIX = "_control"


@dataclass(frozen=True)
class MMMDataset:
    frame: pd.DataFrame
    feature_cols: list[str]
    kpi_col: str
    contribution_cols: list[str]
    channel_keys: list[str]
    train_idx: list[int]
    test_idx: list[int]

    @property
    def X_train(self) -> pd.DataFrame:
        return self.frame.loc[self.train_idx, self.feature_cols]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.frame.loc[self.test_idx, self.feature_cols]

    @property
    def y_train(self) -> pd.Series:
        return self.frame.loc[self.train_idx, self.kpi_col]

    @property
    def y_test(self) -> pd.Series:
        return self.frame.loc[self.test_idx, self.kpi_col]


def _pick_kpi(df: pd.DataFrame, kpi: str | None) -> str:
    if kpi is not None:
        if kpi not in df.columns:
            raise ValueError(f"KPI column {kpi!r} not in data")
        return kpi
    for cand in KPI_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(f"No KPI column among {KPI_CANDIDATES}")


def infer_feature_cols(df: pd.DataFrame, channel_keys: Sequence[str] | None = None) -> list[str]:
    keys = list(channel_keys or CHANNEL_KEYS)
    cols: list[str] = []
    for key in keys:
        for suffix in ("_spend", "_impression"):
            name = f"{key}{suffix}"
            if name in df.columns:
                cols.append(name)
    # Controls
    cols.extend(c for c in df.columns if c.endswith(CONTROL_SUFFIX))
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    if not ordered:
        raise ValueError("No media/control feature columns found")
    return ordered


def infer_contribution_cols(df: pd.DataFrame, channel_keys: Sequence[str] | None = None) -> list[str]:
    keys = list(channel_keys or CHANNEL_KEYS)
    return [f"contribution_{k}" for k in keys if f"contribution_{k}" in df.columns]


def time_split_indices(n: int, train_frac: float = 2 / 3) -> tuple[list[int], list[int]]:
    """Earlier weeks = context/train; later weeks = holdout (TabFM ICL-friendly)."""
    if n < 4:
        raise ValueError("Need at least 4 rows for a meaningful time split")
    n_train = max(2, min(n - 2, int(round(n * train_frac))))
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n))
    return train_idx, test_idx


def load_mmm_dataset(
    path: str | Path | None = None,
    *,
    kpi: str | None = None,
    train_frac: float = 2 / 3,
    channel_keys: Sequence[str] | None = None,
) -> MMMDataset:
    csv_path = Path(path) if path else DEFAULT_CSV
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    keys = list(channel_keys or CHANNEL_KEYS)
    present_keys = [k for k in keys if f"{k}_spend" in df.columns or f"{k}_impression" in df.columns]
    if not present_keys:
        # Fall back: ChannelN_spend style from Meridian sample CSVs
        present_keys = sorted(
            {c[: -len("_spend")] for c in df.columns if c.endswith("_spend")},
            key=lambda s: s,
        )
    feature_cols = infer_feature_cols(df, present_keys)
    contribution_cols = infer_contribution_cols(df, present_keys)
    kpi_col = _pick_kpi(df, kpi)
    train_idx, test_idx = time_split_indices(len(df), train_frac=train_frac)
    return MMMDataset(
        frame=df,
        feature_cols=feature_cols,
        kpi_col=kpi_col,
        contribution_cols=contribution_cols,
        channel_keys=present_keys,
        train_idx=train_idx,
        test_idx=test_idx,
    )
