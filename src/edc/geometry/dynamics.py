"""Descent-dynamics features from the energy/gradient trajectories. [Phase 2]

Steps-to-convergence (first step with ``||grad|| < grad_tol``), monotonicity fraction
(fraction of steps with ΔE < 0), area under the energy curve, and terminal gradient norm
(residual stationarity). Slow, non-monotone, high-residual descents => hypothesised unreliable.
"""

from __future__ import annotations


def dynamics_features(traj, grad_tol: float):
    raise NotImplementedError(
        "Phase 2: steps-to-converge, monotonicity fraction, AUC, terminal residual."
    )
