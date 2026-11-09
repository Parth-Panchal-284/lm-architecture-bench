#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source scripts/env.sh

python scripts/check_env.py
python -m pytest "$NANOLM_ROOT/nanolm/tests" -q
rm -rf outputs/nanolm/smoke
python -m nanolm.train -c configs/smoke.yaml

echo "[smoke] success"
