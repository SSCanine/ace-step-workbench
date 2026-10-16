#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./install.sh
exec .venv/bin/python -m workbench --open "$@"
