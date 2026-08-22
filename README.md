# TabFM vs Meridian — honest gap analysis (official demo data)

> **TabFM predicts KPI; it does not replace Meridian for contribution / ROI / curves / budget opt.**

Phone one-pager for annual budget / strategy planners:
[`docs/decisioning_checklist.md`](docs/decisioning_checklist.md).

That is the practitioner framing for this repo: predictive fit ≠ media decisioning.
This is an **honest gap analysis** on official Meridian simulated/demo data — not a
bake-off framed to crown a winner.

> **Simulated / demo data only.** Official Meridian CSVs here are not estimates of
> real campaign performance. Tiny MCMC runs are **directional / not decision-grade**.

## Fair OOS predictive metrics (after the framing above)

Both models use the **same frozen input table** and the **same later-week holdout**.
Headline comparison is holdout KPI only (RMSE / MAE / R² / MAPE). See
`results/metrics.json` for the latest fair OOS numbers.

| Question | Answer |
| --- | --- |
| Can TabFM match Meridian on **holdout KPI prediction**? | **Yes** (predictive scores only) |
| Can TabFM produce Meridian **contribution / ROI / curves / budget / geo decisioning**? | **No** (Meridian-only product surface) |

Optional `--ablation-proxy` for TabFM is a sensitivity hack — **not** Meridian-equivalent,
**not** causal, and **excluded** from the headline metrics table.

## Fair predictive evaluation (how the OOS numbers are produced)

Both models use the **same frozen input table** and the **same later-week holdout**:

1. Load one official Meridian-schema CSV
2. Build shared `holdout_id` / holdout times (earlier weeks train, later weeks holdout)
3. **TabFM:** ICL context = train rows (optionally capped); predict holdout KPI
4. **Meridian:** `ModelSpec(holdout_id=...)` excludes holdout **KPI** from training; score
   holdout predictive KPI via `expected_outcome` on those weeks (+ `predictive_accuracy` Train/Test)

**Documented remaining asymmetry:** Meridian still uses holdout **media** for Adstock
carryover (upstream design). TabFM never sees holdout rows as context. MCMC here uses
tiny chains — directional PoC, not decision-grade.

## Deliverables matrix

| Meridian deliverable | TabFM? |
| --- | --- |
| Predictive KPI / expected outcome | **Yes** |
| Channel contribution (incremental) | **No** |
| ROI / effectiveness | **No** |
| Response curves | **No** |
| Budget optimization | **No** |
| Geo-level media decisioning | **No** |

Full matrix: `results/deliverables_matrix.md` and `mmm_compare/deliverables.py`.

## Data

| Role | File | Notes |
| --- | --- | --- |
| **Preferred official** (Getting Started) | `data/official/geo_all_channels.csv` | 40 geos × 156 weeks |
| **Runtime default (CPU PoC)** | `data/official/national_all_channels.csv` | Same Meridian schema; tractable MCMC |
| Optional | `hypothetical_geo_all_channels.csv`, `data/legacy/` | Alternate / older dummy |

Pinned URLs + SHA-256: `data/official/SOURCE.md`. Column roles (KPI / media / controls):
`data/SCHEMA.md`.

```bash
# Preferred geo file (Meridian geo fit may skip on CPU)
python scripts/run_comparison.py --dataset geo -v

# Runtime default
python scripts/run_comparison.py --dataset national -v
```

## Quick start

```bash
bash scripts/install_deps.sh
INSTALL_MERIDIAN=1 bash scripts/install_deps.sh

python scripts/run_comparison.py --dry-run
python scripts/run_comparison.py -v
# Optional, quarantined:
python scripts/run_comparison.py -v --ablation-proxy
```

Outputs: `results/metrics.json` (fair OOS KPI), `results/deliverables_matrix.md`,
Meridian-only ROI/contribution tables inside the JSON (not peer-scored against TabFM).

## Models

- **TabFM:** [google-research/tabfm](https://github.com/google-research/tabfm) —
  weights `google/tabfm-1.0.0-pytorch`, license **`tabfm-non-commercial-v1.0`**
  (non-commercial). Optional `HF_TOKEN` — do not commit secrets.
- **Meridian:** [`google-meridian`](https://developers.google.com/meridian) with
  `holdout_id`; fallback Meridian-style Ridge+adstock+Hill if install/fit fails.

## What is / is not claimed

| Claimed | Not claimed |
| --- | --- |
| Fair holdout KPI comparison on frozen official demo data | That TabFM replaces causal MMM |
| Honest deliverables gap analysis | Bake-off “winner” for media decisions |
| Directional Meridian Analyzer tables when fit succeeds | Decision-grade MCMC / real campaign ROI |

## License notes

PoC code Apache-2.0 unless noted; TabFM weights non-commercial; Meridian per upstream terms.
