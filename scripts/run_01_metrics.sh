#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/env.sh

python -m nanolm.sweep \
  --sweep configs/sweeps/untrained_tokenizers.yaml \
  "$@"

python scripts/compare_untrained.py \
  outputs/nanolm/sweeps/untrained_tokenizers \
  --out results/metrics/untrained_tokenizers.csv

echo "[metrics] near-untrained tokenizer check complete"
