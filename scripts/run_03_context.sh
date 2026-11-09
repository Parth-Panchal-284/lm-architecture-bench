#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/env.sh

SWEEP_DIR="outputs/nanolm/sweeps/context_length"
python -m nanolm.sweep \
  --sweep configs/sweeps/context_length.yaml \
  "$@"

python -m nanolm.analyze "$SWEEP_DIR" --metric val_bpb

mkdir -p results/context figures
cp "$SWEEP_DIR/summary.csv" results/context/summary.csv
cp "$SWEEP_DIR/summary.md" results/context/summary.md
python scripts/plot_context.py "$SWEEP_DIR" \
  --csv results/context/context_summary.csv \
  --figure figures/context_length_bpb.png

echo "[context] analysis artifacts copied to results/ and figures/"
