# Official Google Meridian simulated / dummy CSVs

Source (upstream):
https://github.com/google/meridian/tree/main/meridian/data/simulated_data/csv

Pinned raw URLs used to vendor these files:

| File | Raw URL |
| --- | --- |
| `geo_all_channels.csv` | https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv |
| `national_all_channels.csv` | https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/national_all_channels.csv |
| `hypothetical_geo_all_channels.csv` | https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/hypothetical_geo_all_channels.csv |

SHA-256 (at vendoring time):

```
d9ee016f7cd21f5c90b50da10a794af91a372b69f881d3caa266edd41fdafbf6  geo_all_channels.csv
d001b91e494a90d9c88923df6cbd15fd73af132c3609c424daaea3c57c69803a  national_all_channels.csv
97f9c9242166a32a3bfcf176997fdba1943e2e39039ec8f54e7c20d81578d177  hypothetical_geo_all_channels.csv
```

Re-download:

```bash
bash scripts/fetch_official_meridian_data.sh
```

**Disclaimer:** These are Google Meridian *simulated/demo* datasets, not real
campaign performance.

## Runtime default in this PoC

- **Preferred official demo file:** `geo_all_channels.csv` (Getting Started;
  40 geos × 156 weeks).
- **Default for Meridian MCMC on CPU VMs:** `national_all_channels.csv`
  (156 national weeks). Full geo MCMC is typically too slow / memory-heavy
  without a GPU. Pass `--dataset geo` to attempt the geo model, or
  `--dataset geo-agg` to nationally aggregate geo rows for TabFM-style
  prediction while still documenting that Meridian geo insights need a geo fit.

Legacy hand-rolled HK skincare dummy lives under `data/legacy/` (optional).
