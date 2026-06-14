#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT_DIR="${1:?Usage: $0 /path/to/checkpoint_dir}"

python predict.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_csv predictions.csv
