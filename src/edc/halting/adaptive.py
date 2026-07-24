"""Geometry-driven adaptive halting policy. [Phase 3]

Stop the descent as soon as a geometry-based certificate (e.g. basin agreement / curvature
crossing a CRC-calibrated threshold) says the answer is settled, saving compute versus a fixed
step budget — with a guaranteed error budget from ``edc.conformal.crc``. Produces the
compute-vs-accuracy Pareto (figure F4).
"""

from __future__ import annotations


def halting_policy(traj, threshold):
    raise NotImplementedError("Phase 3: per-input stop step from geometry crossing a threshold.")
