# Results

`metrics.json` / `metrics.csv` are from a successful cloud-agent run with:

- **TabFM** mode `tabfm` (PyTorch weights `google/tabfm-1.0.0-pytorch`)
- **Meridian** mode `meridian` (google-meridian 1.8.0, minimal MCMC)

All numbers are on **synthetic** `data/hk_skincare_mmm_dummy.csv` only.

Re-run locally:

```bash
python scripts/run_comparison.py -v
# or
python scripts/run_comparison.py --dry-run
```
