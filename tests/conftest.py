"""Shared test configuration: force JAX onto CPU so the suite never needs a GPU."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
