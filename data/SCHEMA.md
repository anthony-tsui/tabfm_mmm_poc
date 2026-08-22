# Data layout

## Official (primary)

One frozen Meridian-schema input per run under `data/official/` — Google Meridian
**simulated/demo** CSVs. See `data/official/SOURCE.md` for pinned URLs + SHA-256.

| CLI `--dataset` | File | Role |
| --- | --- | --- |
| `geo` | `geo_all_channels.csv` | **Preferred** Getting Started official file (40×156) |
| `national` | `national_all_channels.csv` | **Runtime default** on CPU (tractable Meridian MCMC) |
| `geo-agg` | aggregates `geo_all_channels.csv` | National sums for tabular prediction |
| `hypothetical-geo` | `hypothetical_geo_all_channels.csv` | Shorter geo panel |

### Column roles (national / geo)

| Role | Columns |
| --- | --- |
| **Time / geo keys** | `time`; geo also has `geo`, `population` |
| **KPI** | `conversions` (primary in this PoC); `revenue_per_conversion` |
| **Paid media** | `Channel{0..4}_impression`, `Channel{0..4}_spend` |
| **Organic media** | `Organic_channel0_impression` |
| **Controls** | `competitor_sales_control`, `sentiment_score_control` |
| **Non-media treatment** | `Promo` |

TabFM features = paid media + organic + controls + Promo. Target = KPI (`conversions`).

**Disclaimer:** Simulated/demo only — not real campaign performance.

## Legacy (optional)

`data/legacy/` — earlier hand-rolled HK dummy (`--dataset legacy`). Not the default.
