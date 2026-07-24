"""Shared figure style + one function per paper figure.

Figures are produced ONLY here and called from ``analysis/make_figures.py``, which reads
the ledger and never re-runs training (invariant 6). matplotlib is an optional dependency
(``--extra plot``); importing this module without it fails only when a plot is requested.

Phase 2+ fills in the concrete F1..F6 functions listed in ``paper/README.md``.
"""

from __future__ import annotations

from typing import Any

STYLE = {
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def use_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)


def risk_coverage_curve(rows: list[dict[str, Any]], out_path: str) -> str:  # F2
    """Risk-coverage curves per nonconformity score. Implemented in Phase 3."""
    raise NotImplementedError("Phase 3: risk-coverage figure (F2).")
