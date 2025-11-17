#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
pip install -r requirements.txt
# Ensure system dependencies and browser binaries are installed inside the container
python3 -m playwright install-deps chromium >/dev/null 2>&1 || true
python3 -m playwright install chromium
