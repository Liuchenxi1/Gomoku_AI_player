#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
".venv/bin/python" train.py \
  --resume checkpoints/15x15_500_trial.pt \
  --simulations 500 \
  --games-per-iteration 20 \
  --iterations 100 \
  --updates 30 \
  --batch-size 64 \
  --device auto \
  --output checkpoints/15x15_500.pt
