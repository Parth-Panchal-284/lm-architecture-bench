#!/usr/bin/env bash
# Source this file from the project root before direct nanolm commands.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
UPSTREAM_ROOT="$WORKSPACE_ROOT/_deps/DeepDataMiningLearning"
NANOLM_ROOT="$UPSTREAM_ROOT/DeepDataMiningLearning/llm"

if [[ ! -d "$NANOLM_ROOT/nanolm" ]]; then
  echo "[env] nanolm dependency not found at: $NANOLM_ROOT/nanolm" >&2
  echo "[env] run: bash scripts/bootstrap_upstream.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT WORKSPACE_ROOT UPSTREAM_ROOT NANOLM_ROOT
export PYTHONPATH="$NANOLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
