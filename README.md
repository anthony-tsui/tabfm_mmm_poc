# TabFM vs Meridian — planner examples (official demo data)

> **TabFM predicts KPI; it does not replace Meridian for contribution / ROI / curves / budget opt.**

## Start here (phone)

**Can TabFM be an MMM alternative for annual marketing budget & strategy?**  
→ **No.** See the worked example:

👉 **[`docs/examples_for_planners.md`](docs/examples_for_planners.md)**  
👉 Images: [`examples/01_meridian_annual_mix_story.png`](examples/01_meridian_annual_mix_story.png) · [`examples/02_tabfm_kpi_stops_here.png`](examples/02_tabfm_kpi_stops_here.png)

On this simulated run: **Meridian says cut Channel3 / scale Channel2** (ROI ~53 vs ~104).  
**TabFM only forecasts holdout KPI** and stops — no mix recommendation.

Checklist one-pager: [`docs/decisioning_checklist.md`](docs/decisioning_checklist.md)

> **Simulated / demo data only.** Tiny MCMC is **directional / not decision-grade**. We do not fake TabFM budget advice.

## What this repo is

An honest gap analysis on official Meridian simulated CSVs: one frozen national slice, one planner story with pictures — not a bake-off of R² tables.

| Role | Path |
| --- | --- |
| Preferred official file | `data/official/geo_all_channels.csv` (`--dataset geo`) |
| Runtime default (CPU) | `data/official/national_all_channels.csv` |
| Column roles | `data/SCHEMA.md` |

```bash
bash scripts/install_deps.sh
INSTALL_MERIDIAN=1 bash scripts/install_deps.sh
python scripts/generate_planner_examples.py   # rebuild phone PNGs
python scripts/run_comparison.py --dry-run    # optional technical path
```

## Secondary: technical fair OOS metrics

Fair holdout KPI scoring (shared `holdout_id`) lives under `results/` and `mmm_compare/`. Useful for engineers; **not** the planner headline. Ablation proxies are opt-in (`--ablation-proxy`) and are **not** Meridian contribution.

## License notes

TabFM pretrained weights: non-commercial (`tabfm-non-commercial-v1.0`). Meridian / simulated data: upstream terms.
