"""PURE RGB experimental adapters."""

from pathlib import Path
import os
import sys

WORKSPACE_ROOT = Path(os.environ.get("PURE_RGB_TEST_WORKSPACE", Path(__file__).resolve().parents[1]))

# Resolve the pinned source directly; editable .pth loading varies across runtimes.
_negpy_source = str(WORKSPACE_ROOT / "vendor/negpy")
if _negpy_source not in sys.path:
    sys.path.insert(0, _negpy_source)
