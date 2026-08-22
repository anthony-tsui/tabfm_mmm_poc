"""Load Meridian official simulated CSVs (and optional legacy dummy data)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "data" / "official"
LEGACY_DIR = ROOT / "data" / "legacy"

DATASET_FILES = {
    "geo": OFFICIAL_DIR / "geo_all_channels.csv",
    "national": OFFICIAL_DIR / "national_all_channels.csv",
    "geo-agg": OFFICIAL_DIR / "geo_all_channels.csv",
    "hypothetical-geo": OFFICIAL_DIR / "hypothetical_geo_all_channels.csv",
    "legacy": LEGACY_DIR / "hk_skincare_mmm_dummy.csv",
}

DatasetName = Literal["geo", "national", "geo-agg", "hypothetical-geo", "legacy"]

# Preferred official Getting Started file; CPU Meridian default is national.
PREFERRED_OFFICIAL = "geo"
DEFAULT_RUNTIME_DATASET: DatasetName = "national"

KPI_CANDIDATES = ("conversions", "revenue")
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
    dataset_name: str
    is_geo: bool
    source_path: str
    notes: str = ""

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


def resolve_dataset_path(dataset: str | None = None, path: str | Path | None = None) -> tuple[Path, str]:
    if path is not None:
        p = Path(path)
        return p, p.name
    name = dataset or DEFAULT_RUNTIME_DATASET
    if name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASET_FILES)}")
    return DATASET_FILES[name], name


def _channel_keys_from_spend(df: pd.DataFrame) -> list[str]:
    keys = [c[: -len("_spend")] for c in df.columns if c.endswith("_spend")]
    # Prefer Channel0..ChannelN numeric order when present
    def sort_key(k: str) -> tuple:
        if k.startswith("Channel") and k[7:].isdigit():
            return (0, int(k[7:]))
        return (1, k)

    return sorted(set(keys), key=sort_key)


def infer_feature_cols(df: pd.DataFrame, channel_keys: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for key in channel_keys:
        for suffix in ("_spend", "_impression"):
            name = f"{key}{suffix}"
            if name in df.columns:
                cols.append(name)
    # Organic media + controls + common non-media treatments
    for c in df.columns:
        if c.startswith("Organic_") or c.endswith(CONTROL_SUFFIX) or c in ("Promo", "promo_control"):
            if c not in cols:
                cols.append(c)
    if not cols:
        raise ValueError("No media/control feature columns found")
    return cols


def infer_contribution_cols(df: pd.DataFrame, channel_keys: Sequence[str]) -> list[str]:
    return [f"contribution_{k}" for k in channel_keys if f"contribution_{k}" in df.columns]


def time_split_indices(n: int, train_frac: float = 2 / 3) -> tuple[list[int], list[int]]:
    if n < 4:
        raise ValueError("Need at least 4 rows for a meaningful time split")
    n_train = max(2, min(n - 2, int(round(n * train_frac))))
    return list(range(n_train)), list(range(n_train, n))


def _aggregate_geo_to_national(df: pd.DataFrame) -> pd.DataFrame:
    """Sum media/KPI across geos; average bounded controls."""
    if "geo" not in df.columns:
        return df
    sum_cols = [
        c
        for c in df.columns
        if c.endswith(("_spend", "_impression"))
        or c in ("conversions", "revenue", "population")
        or c.startswith("Organic_")
    ]
    mean_cols = [
        c
        for c in df.columns
        if c.endswith(CONTROL_SUFFIX) or c in ("Promo", "revenue_per_conversion")
    ]
    agg: dict[str, str] = {c: "sum" for c in sum_cols if c in df.columns}
    agg.update({c: "mean" for c in mean_cols if c in df.columns})
    out = df.groupby("time", as_index=False).agg(agg)
    return out.sort_values("time").reset_index(drop=True)


def _prepare_frame(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, bool, str]:
    notes = ""
    is_geo = "geo" in df.columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if dataset_name == "geo-agg" and is_geo:
        df = _aggregate_geo_to_national(df)
        notes = "geo_all_channels nationally aggregated (sum media/KPI) for tabular prediction"
        is_geo = False
    elif is_geo and dataset_name == "geo":
        # Keep panel rows (geo × time) for TabFM; Meridian path handles geo builder separately.
        df = df.sort_values(["time", "geo"]).reset_index(drop=True)
        notes = "geo panel retained (geo × time rows)"
    else:
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)
        notes = "national weekly series"
    return df, is_geo, notes


def _pick_kpi(df: pd.DataFrame, kpi: str | None) -> str:
    if kpi is not None:
        if kpi not in df.columns:
            raise ValueError(f"KPI column {kpi!r} not in data")
        return kpi
    for cand in KPI_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(f"No KPI column among {KPI_CANDIDATES}")


def load_mmm_dataset(
    path: str | Path | None = None,
    *,
    dataset: str | None = None,
    kpi: str | None = None,
    train_frac: float = 2 / 3,
    channel_keys: Sequence[str] | None = None,
    max_context_rows: int | None = None,
) -> MMMDataset:
    """Load official Meridian simulated data (default) or an explicit CSV path.

    Time split uses unique times when a geo panel is present (hold out later weeks
    across all geos). For national / geo-agg, split is on row order after time sort.
    """
    csv_path, dataset_name = resolve_dataset_path(dataset=dataset, path=path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run scripts/fetch_official_meridian_data.sh or check data/official/."
        )
    raw = pd.read_csv(csv_path)
    df, is_geo, notes = _prepare_frame(raw, dataset_name)
    keys = list(channel_keys) if channel_keys else _channel_keys_from_spend(df)
    feature_cols = infer_feature_cols(df, keys)
    # Drop non-numeric / id cols from features if present
    feature_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    contribution_cols = infer_contribution_cols(df, keys)
    kpi_col = _pick_kpi(df, kpi)

    if is_geo and "time" in df.columns:
        times = sorted(df["time"].unique())
        train_idx_t, test_idx_t = time_split_indices(len(times), train_frac=train_frac)
        train_times = {times[i] for i in train_idx_t}
        test_times = {times[i] for i in test_idx_t}
        train_idx = df.index[df["time"].isin(train_times)].tolist()
        test_idx = df.index[df["time"].isin(test_times)].tolist()
        notes += (
            f"; time-split on {len(train_idx_t)}/{len(test_idx_t)} weeks "
            f"(panel rows {len(train_idx)}/{len(test_idx)})"
        )
    else:
        train_idx, test_idx = time_split_indices(len(df), train_frac=train_frac)

    if max_context_rows is not None and len(train_idx) > max_context_rows:
        # Keep most recent train rows as TabFM context (ICL window)
        train_idx = train_idx[-max_context_rows:]
        notes += f"; truncated train context to last {max_context_rows} rows"

    return MMMDataset(
        frame=df,
        feature_cols=feature_cols,
        kpi_col=kpi_col,
        contribution_cols=contribution_cols,
        channel_keys=keys,
        train_idx=train_idx,
        test_idx=test_idx,
        dataset_name=dataset_name,
        is_geo=is_geo,
        source_path=str(csv_path),
        notes=notes,
    )
