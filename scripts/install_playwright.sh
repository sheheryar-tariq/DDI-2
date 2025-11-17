#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
pip install -r requirements.txt
python3 -m playwright install chromium
