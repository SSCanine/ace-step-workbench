#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== ACE-Step Workbench installer ==="
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (fast Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python install.py "$@"
echo "Done. Run ./start.sh to launch."
