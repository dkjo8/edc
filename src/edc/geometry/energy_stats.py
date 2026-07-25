"""Terminal-energy statistics across restarts. [Phase 2]

Mean, min, and spread of the terminal energy over the K restarts. These reproduce the
**scalar-energy / best-of-N** signal that Energy-Based Transformers use — here they are just
features, and the falsification test is whether the geometry features beat them (ΔAURC).

Plain NumPy (invariant 1): JAX stays in ``energy/``, ``inference/`` and ``curvature.py``.
"""

from __future__ import annotations

import numpy as np

from edc.inference.trajectory import TrajectoryRecord


def energy_features(traj: TrajectoryRecord) -> tuple[np.ndarray, list[str]]:
    """``(B, 3)`` mean/min/std of terminal energy across restarts, plus feature names."""
    e = np.asarray(traj.terminal_energy, dtype=float)          # (B, K)
    feats = np.stack([e.mean(axis=1), e.min(axis=1), e.std(axis=1)], axis=1)
    return feats, ["energy/mean", "energy/min", "energy/std"]
