"""Assemble the per-input geometry feature vector from a TrajectoryRecord. [Phase 2]

Combines: basin agreement (``basin``), terminal-energy statistics (``energy_stats``),
curvature (``curvature``), and descent dynamics (``dynamics``) into the nonconformity
feature vector that Phase-3 conformal calibration consumes. This is the paper's contribution.
"""

from __future__ import annotations

from edc.inference.trajectory import TrajectoryRecord


def geometry_features(traj: TrajectoryRecord, fns, params, key):
    """Return ``(B, n_features)`` array + feature-name list. Implemented in Phase 2."""
    raise NotImplementedError(
        "Phase 2: assemble basin + energy_stats + curvature + dynamics features."
    )
