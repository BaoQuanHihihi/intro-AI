#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python train.py \
  --config configs/default.yaml \
  --output_dir outputs/run_$(date +%Y%m%d_%H%M%S)
