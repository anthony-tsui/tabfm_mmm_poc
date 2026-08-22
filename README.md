# TabFM vs Meridian MMM — synthetic / official-demo comparison PoC

Side-by-side evaluation of **Google Research TabFM** (tabular foundation model)
and **Google Meridian** on **official Meridian simulated/demo data**, plus an
honest checklist of whether TabFM can produce Meridian’s actionable deliverables.

> **Simulated / demo data only.** Official Meridian CSVs and any legacy dummy
> files here are not estimates of real campaign performance. Do not use these
> numbers for budgeting or business decisions.

## Headline answer: can TabFM output the same as Meridian?

**Mostly no.** TabFM is a zero-shot tabular ICL **regressor**, not an MMM.

| Capability counts (see matrix below) | |
| --- | ---: |
| Yes | 1 (predictive KPI fit only) |
| Partial | 2 (ablation / geo-row prediction — not Meridian-equivalent) |
| No | 3 (ROI, response curves, budget optimization) |

### Deliverables matrix

| Meridian deliverable | How Meridian produces it | TabFM equivalent? | If Partial / notes |
| --- | --- | --- | --- |
| Expected outcome / predictive fit | `Analyzer.expected_outcome` / `predictive_accuracy` | **Yes** | Comparable as predictive scores only, not as a structural MMM fit |
| Channel contribution (incremental) | `Analyzer.incremental_outcome` | **Partial** | Optional leave-one-channel ablation is a hacky proxy — **not** causal / Meridian-equivalent |
| ROI / effectiveness by channel | `Analyzer.roi` / `summary_metrics` | **No** | No structural ROI |
| Response curves | `Analyzer.response_curves` | **No** | No adstock/Hill response API |
| Budget optimization / scenarios | `meridian.analysis.optimizer` | **No** | No Meridian optimizer |
| Geo-level insights | Geo hierarchical Meridian + Analyzer | **Partial** | Can score geo-week rows; no hierarchical geo posteriors / geo ROI |

The same matrix is exported to `results/deliverables_matrix.md` on each run and
rendered in `tabfm_vs_meridian.ipynb`.

## What this PoC measures

1. **Predictive metrics** (where comparable): holdout RMSE / MAE / R² / MAPE on KPI
2. **Deliverables checklist**: Meridian artifacts extracted when a real fit succeeds
   (contribution, ROI, summary metrics, response-curve preview) vs TabFM gaps
3. Optional contribution recovery only if planted `contribution_*` exist (legacy data)

## Data

- **Preferred official demo:** `data/official/geo_all_channels.csv` (Getting Started
  [`geo_all_channels.csv`](https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv))
- **Runtime default on this CPU PoC:** `data/official/national_all_channels.csv`
  — full 40×156 geo MCMC is typically too slow without a GPU. Documented in
  `data/official/SOURCE.md` (pinned URL + SHA-256).
- Flags: `--dataset national|geo|geo-agg|hypothetical-geo|legacy`
- Legacy hand-rolled HK dummy: `data/legacy/` (optional)

## Models

### TabFM

- [google-research/tabfm](https://github.com/google-research/tabfm) — `TabFMRegressor`
- Weights: Hugging Face `google/tabfm-1.0.0-pytorch`
- **Weight license:** pretrained weights are **`tabfm-non-commercial-v1.0`**
  (non-commercial / non-production). Source is Apache-2.0.
- `--dry-run` uses a Ridge mock so the pipeline always completes
- Optional `HF_TOKEN` for Hub downloads — do not commit tokens

### Meridian

- Prefer real [`google-meridian`](https://developers.google.com/meridian) with
  **minimal** MCMC; extract Analyzer deliverables when possible
- Fallback: labeled **Meridian-style** adstock + Hill + Ridge proxy (not Bayesian Meridian)

## Quick start

```bash
bash scripts/install_deps.sh
INSTALL_MERIDIAN=1 bash scripts/install_deps.sh   # optional

# Always-works path
python scripts/run_comparison.py --dry-run

# Default: official national_all_channels + best-effort TabFM/Meridian
python scripts/run_comparison.py -v

# Attempt official geo file (Meridian geo fit may skip on CPU)
python scripts/run_comparison.py --dataset geo -v

# Nationally aggregate geo CSV for tabular prediction
python scripts/run_comparison.py --dataset geo-agg --no-meridian -v
```

Outputs: `results/metrics.json`, `results/deliverables_matrix.md`, ROI/contribution
tables inside the JSON / notebook when Meridian succeeds.

## Layout

```text
data/official/        vendored Meridian simulated CSVs + SOURCE.md
data/legacy/          optional older hand-rolled dummy
mmm_compare/          load, TabFM, Meridian, deliverables matrix, metrics
scripts/              fetch data, install, CLI
results/              metrics + deliverables matrix from a run
tabfm_vs_meridian.ipynb
```

## Assumptions

1. Default dataset is **national** for CPU Meridian tractability; geo CSV is vendored
   as the preferred official Getting Started file.
2. TabFM context capped at 100 rows (`--max-context-rows`) per upstream ICL limits.
3. Real Meridian holdout KPI uses in-sample `expected_outcome` on later weeks
   (fit on full series). TabFM is true OOS ICL on the time split.
4. Tiny MCMC settings are PoC-only — not production Meridian quality.
5. We do **not** fake Meridian ROI / response curves / budget opt from TabFM.

## What is / is not claimed

| Claimed | Not claimed |
| --- | --- |
| Runnable comparison on official Meridian demo data | Real campaign ROI / budgets |
| Honest deliverables gap analysis | That TabFM replaces causal MMM |
| Best-effort Meridian Analyzer tables when fit succeeds | Production-calibrated Meridian chains |

## License notes

- This PoC code: Apache-2.0 unless otherwise noted
- TabFM pretrained weights: non-commercial (upstream)
- Meridian + its simulated data: follow Google Meridian / package terms
