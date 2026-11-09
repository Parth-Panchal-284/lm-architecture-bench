#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/env.sh

RUN_DIR="${1:-outputs/nanolm/sweeps/architecture/gpt_modern_s1337}"
if [[ ! -f "$RUN_DIR/model.pt" ]]; then
  echo "[attention] checkpoint not found: $RUN_DIR/model.pt" >&2
  echo "[attention] run the architecture sweep first or pass another Transformer run directory." >&2
  exit 1
fi

mkdir -p figures results/attention

DIALOGUE=$'ROMEO:\nBut soft! what light through yonder window breaks?'
PROSE='The quick brown fox jumps over the lazy dog while morning light fills the room.'
STRUCTURED=$'Name: Alice\nRole: Engineer\nName: Bob\nRole: Researcher'

python -m nanolm.viz attention --run "$RUN_DIR" --text "$DIALOGUE"
cp docs/nanolm_figs/p1_attention_maps.png figures/attention_dialogue.png

python -m nanolm.viz attention --run "$RUN_DIR" --text "$PROSE"
cp docs/nanolm_figs/p1_attention_maps.png figures/attention_prose.png

python -m nanolm.viz attention --run "$RUN_DIR" --text "$STRUCTURED"
cp docs/nanolm_figs/p1_attention_maps.png figures/attention_structured.png

python scripts/attention_metrics.py \
  --run "$RUN_DIR" \
  --texts references/attention_texts.json \
  --csv results/attention/head_metrics.csv \
  --summary results/attention/summary.md

echo "[attention] figures and numerical head metrics are ready"
