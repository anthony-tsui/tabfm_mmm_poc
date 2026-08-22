# Data layout

## Official (default)

`data/official/` — Google Meridian **simulated/demo** CSVs vendored from
https://github.com/google/meridian/tree/main/meridian/data/simulated_data/csv

See `data/official/SOURCE.md` for pinned URLs and SHA-256 checksums.

| File | Grain | Typical use in this PoC |
| --- | --- | --- |
| `geo_all_channels.csv` | 40 geos × 156 weeks | Preferred Getting Started file; geo MCMC usually too heavy on CPU |
| `national_all_channels.csv` | 156 national weeks | **Runtime default** for Meridian fit + TabFM comparison |
| `hypothetical_geo_all_channels.csv` | shorter geo panel | Optional |

**Disclaimer:** Simulated/demo values only — not real campaign performance.

### Common columns (national / geo)

- `time`, `conversions`, `revenue_per_conversion`
- `Channel{0..4}_impression`, `Channel{0..4}_spend`
- `Organic_channel0_impression`
- Controls: `competitor_sales_control`, `sentiment_score_control`
- Non-media: `Promo`
- Geo-only: `geo`, `population`

## Legacy (optional)

`data/legacy/hk_skincare_mmm_dummy.csv` — earlier hand-rolled 12-week dummy with
planted `contribution_*` columns. Not the default. Use `--dataset legacy`.
