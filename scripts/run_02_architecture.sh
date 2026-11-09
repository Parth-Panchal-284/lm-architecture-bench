#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/env.sh

SWEEP_DIR="outputs/nanolm/sweeps/architecture"
python -m nanolm.sweep \
  --sweep configs/sweeps/architecture.yaml \
  "$@"

python -m nanolm.analyze "$SWEEP_DIR" \
  --baseline gpt_textbook \
  --metric val_bpb

mkdir -p results/architecture figures
cp "$SWEEP_DIR/summary.csv" results/architecture/summary.csv
cp "$SWEEP_DIR/summary.md" results/architecture/summary.md
cp "$SWEEP_DIR/curves.png" figures/architecture_learning_curves.png
cp "$SWEEP_DIR/bars.png" figures/architecture_final_bpb.png

TRAINED_RUN="$SWEEP_DIR/gpt_modern_s1337/metrics.json"
python scripts/metrics_by_hand.py "$TRAINED_RUN" \
  --out results/metrics/trained_char_metrics.json

echo "[architecture] analysis artifacts copied to results/ and figures/"
