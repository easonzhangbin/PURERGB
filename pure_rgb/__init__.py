"""PURE RGB experimental adapters."""

from pathlib import Path
import os
import sys

WORKSPACE_ROOT = Path(os.environ.get("PURE_RGB_TEST_WORKSPACE", Path(__file__).resolve().parents[1]))

BUNDLED_VENDOR = os.environ.get("PURE_RGB_BUNDLED_VENDOR")
VENDOR_ROOT = Path(BUNDLED_VENDOR) if BUNDLED_VENDOR else WORKSPACE_ROOT / "vendor"

# Resolve the pinned source directly; editable .pth loading varies across runtimes.
_negpy_source = str(VENDOR_ROOT / "negpy")
if _negpy_source not in sys.path:
    sys.path.insert(0, _negpy_source)
