#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Python 3 is missing. Install Python 3.11+ from python.org or Homebrew.' >&2
  exit 1
fi
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -c 'import torch; print("Torch:", torch.__version__, "MPS available:", torch.backends.mps.is_available())'
echo 'Setup complete. Activate with: source .venv/bin/activate'
