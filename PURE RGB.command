#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
exec "$ROOT/vendor/negpy/.venv/bin/python" -m pure_rgb.workflow_launcher "$@"
