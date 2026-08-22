#!/usr/bin/env bash
# Best-effort dependency install for the TabFM vs Meridian PoC.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m pip install -U pip wheel setuptools
python3 -m pip install -r requirements.txt

# TabFM from Google Research (PyTorch backend)
if [[ ! -d vendor/tabfm ]]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/google-research/tabfm.git vendor/tabfm
fi
python3 -m pip install -e "vendor/tabfm[pytorch]" || {
  echo "WARN: TabFM editable install failed; dry-run mock path remains available."
}

# Optional Meridian (may pull tensorflow / jax stack)
if [[ "${INSTALL_MERIDIAN:-0}" == "1" ]]; then
  python3 -m pip install "google-meridian" || {
    echo "WARN: google-meridian install failed; Meridian-style proxy will be used."
  }
fi

echo "Done. Try: python scripts/run_comparison.py --dry-run"
