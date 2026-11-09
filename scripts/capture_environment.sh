#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/env.sh

mkdir -p results/environment
python scripts/check_env.py | tee results/environment/system.txt
python -m pip freeze > results/environment/pip_freeze.txt
git -C "$UPSTREAM_ROOT" rev-parse HEAD > results/environment/upstream_commit.txt
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > results/environment/project_commit.txt || true
fi
