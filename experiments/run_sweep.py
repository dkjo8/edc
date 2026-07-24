"""Expand a sweep grid into many runs. [Phase 4]

    PYTHONPATH=src python experiments/run_sweep.py configs/sweeps/<grid>.toml

Grids sweep K restarts, step size, Langevin temperature, and conformal alpha (see
configs/sweeps/). Each cell calls the same path as run_experiment and appends its own row.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    raise NotImplementedError("Phase 4: cartesian-expand the grid and dispatch run_experiment.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
