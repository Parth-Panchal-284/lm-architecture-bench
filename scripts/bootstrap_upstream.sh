#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"
DEP_ROOT="$WORKSPACE_ROOT/_deps/DeepDataMiningLearning"
PIN_FILE="$PROJECT_ROOT/references/upstream_commit.txt"
UPSTREAM_URL="https://github.com/lkk688/DeepDataMiningLearning.git"

mkdir -p "$WORKSPACE_ROOT/_deps"

if [[ ! -d "$DEP_ROOT/.git" ]]; then
  echo "[bootstrap] cloning reference repository to:"
  echo "  $DEP_ROOT"
  git clone "$UPSTREAM_URL" "$DEP_ROOT"
fi

PIN="$(tr -d '[:space:]' < "$PIN_FILE")"
if [[ "$PIN" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "[bootstrap] checking out pinned upstream commit: $PIN"
  git -C "$DEP_ROOT" fetch origin "$PIN" || true
  git -C "$DEP_ROOT" checkout --detach "$PIN"
else
  echo "[bootstrap] no commit pinned yet; resolving current origin/main"
  git -C "$DEP_ROOT" fetch origin main
  git -C "$DEP_ROOT" checkout main
  git -C "$DEP_ROOT" pull --ff-only origin main
  RESOLVED="$(git -C "$DEP_ROOT" rev-parse HEAD)"
  printf '%s\n' "$RESOLVED" > "$PIN_FILE"
  echo "[bootstrap] resolved upstream commit: $RESOLVED"
  echo "[bootstrap] after the smoke test succeeds, commit references/upstream_commit.txt"
fi

NANOLM_ROOT="$DEP_ROOT/DeepDataMiningLearning/llm"
if [[ ! -d "$NANOLM_ROOT/nanolm" ]]; then
  echo "[bootstrap] ERROR: expected nanolm package not found at:"
  echo "  $NANOLM_ROOT/nanolm" >&2
  exit 1
fi

echo "[bootstrap] nanolm package: $NANOLM_ROOT/nanolm"
