#!/usr/bin/env bash
# Re-fetch official Meridian simulated CSVs into data/official/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/official"
mkdir -p "$OUT"
BASE="https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv"
for f in geo_all_channels.csv national_all_channels.csv hypothetical_geo_all_channels.csv; do
  echo "Fetching $f ..."
  curl -fsSL -o "$OUT/$f" "$BASE/$f"
done
echo "SHA-256:"
sha256sum "$OUT"/*.csv
echo "Done. See data/official/SOURCE.md"
