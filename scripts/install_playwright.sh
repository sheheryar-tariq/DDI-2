#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Make sure browser binaries live in a path writable at runtime
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/app/.cache/ms-playwright}"
mkdir -p "$PLAYWRIGHT_BROWSERS_PATH"
# Ensure system dependencies and browser binaries are installed inside the container
python3 -m playwright install-deps chromium >/dev/null 2>&1 || true
python3 -m playwright install chromium
