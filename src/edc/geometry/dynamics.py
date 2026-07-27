"""Descent-dynamics features from the energy/gradient trajectories. [Phase 2]

Steps-to-convergence (first step with ``||grad|| < grad_tol``), monotonicity fraction
(fraction of steps with ΔE < 0), total energy drop, and terminal gradient norm (residual
stationarity). Slow, non-monotone, high-residual descents => hypothesised unreliable.

Computed per restart from the ``(B, K, ...)`` curves, then averaged over the K restarts to a
per-input ``(B, ...)`` value. Plain NumPy (invariant 1).
"""

from __future__ import annotations

import numpy as np

from edc.inference.trajectory import TrajectoryRecord


def dynamics_features(traj: TrajectoryRecord, grad_tol: float) -> tuple[np.ndarray, list[str]]:
    """``(B, 4)`` descent-dynamics features (mean over restarts) plus names."""
    energies = np.asarray(traj.energies, dtype=float)          # (B, K, T+1)
    gnorms = np.asarray(traj.grad_norms, dtype=float)          # (B, K, T)
    T = gnorms.shape[-1]

    # Steps-to-converge: first step under grad_tol, else T. Normalise by T to keep it in [0, 1].
    below = gnorms < grad_tol                                  # (B, K, T)
    any_below = below.any(axis=-1)
    first = below.argmax(axis=-1)                              # 0 when none are below
    steps = np.where(any_below, first, T).astype(float) / T

    # Monotonicity: fraction of steps where energy decreased.
    de = np.diff(energies, axis=-1)                            # (B, K, T)
    monotonic = (de < 0).mean(axis=-1)

    # Total energy drop from the initial point to the endpoint.
    drop = energies[..., 0] - energies[..., -1]               # (B, K)

    # Terminal residual: last-step gradient norm (stationarity proxy at ~z*).
    residual = np.asarray(traj.terminal_grad_norm, dtype=float)  # (B, K)

    feats = np.stack(
        [steps.mean(axis=1), monotonic.mean(axis=1), drop.mean(axis=1), residual.mean(axis=1)],
        axis=1,
    )
    return feats, ["dynamics/steps", "dynamics/monotonic", "dynamics/drop", "dynamics/residual"]
