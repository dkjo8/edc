"""Shared test configuration: force JAX onto CPU so the suite never needs a GPU."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The experiments/ and analysis/ scripts are not a package; expose them for direct import in tests
# (the same way they resolve sibling imports when run as scripts).
_ROOT = Path(__file__).resolve().parents[1]
for _d in ("experiments", "analysis"):
    sys.path.insert(0, str(_ROOT / _d))
