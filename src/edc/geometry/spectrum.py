"""Full-Hessian-spectrum features at the inference optimum. [Phase 4k, richer geometry]

Phase 2's curvature group summarises the latent Hessian by only ``lambda_max`` (power iteration)
and ``tr(H)`` (Hutchinson). That is a thin description of the basin: it says nothing about the
*smallest* eigenvalue (is ``z*`` a genuine minimum or a saddle?), the basin's *anisotropy*, or a
fuller volume proxy than the trace. This module takes the **exact** eigenvalue spectrum of the
``d x d`` latent Hessian (``d`` is small) and derives four richer descriptors, each summarised
mean-over-restarts and at-the-best (min-energy) restart to match the ``_mean``/``_best`` convention.

Plain NumPy assembler (invariant 1): the only JAX is behind ``curvature.batched_spectrum``, whose
eigenvalues are immediately pulled back to NumPy here.
"""

from __future__ import annotations

import numpy as np

from edc.geometry import curvature
from edc.inference.trajectory import TrajectoryRecord

_EPS = 1e-8


def _mean_and_best(per_particle: np.ndarray, best: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a ``(B, K)`` per-restart quantity to (mean over K, value at the best restart)."""
    rows = np.arange(per_particle.shape[0])
    return per_particle.mean(axis=1), per_particle[rows, best]


def spectrum_features(traj: TrajectoryRecord, fns, params) -> tuple[np.ndarray, list[str]]:
    """``(B, 8)`` full-spectrum features plus names.

    Per particle, from the ascending eigenvalues ``lambda`` of ``H = d^2/dz^2 E(h_x, z*)``:

    * ``spectrum/lmin_*``    — smallest eigenvalue (negative => ``z*`` is a saddle, not a minimum).
    * ``spectrum/negfrac_*`` — fraction of eigenvalues ``< -eps`` (how non-minimal the basin is).
    * ``spectrum/effrank_*`` — participation ratio ``(sum lambda_+)^2 / sum lambda_+^2`` over the
      positive eigenvalues (basin isotropy/anisotropy — trace alone cannot express this).
    * ``spectrum/logdet_*``  — ``sum log(lambda_+ + eps)`` over positive eigenvalues (a fuller
      basin-volume proxy than ``tr(H)``).

    Each is reported mean-over-restarts and at the min-energy restart (``*_mean``/``*_best``).
    """
    z = np.asarray(traj.z_star, dtype=float)                   # (B, K, d)
    B, K, d = z.shape
    h_x = np.asarray(traj.h_x, dtype=float)                    # (B, dc)

    # Flatten to N = B*K particles exactly as inference.restarts lays them out.
    contexts = np.repeat(h_x, K, axis=0)                       # (N, dc)
    zs = z.reshape(B * K, d)                                   # (N, d)

    evals = np.asarray(curvature.batched_spectrum(fns.energy, params, contexts, zs), dtype=float)
    evals = evals.reshape(B, K, d)                             # (B, K, d) ascending

    lmin = evals[..., 0]                                       # (B, K)
    negfrac = (evals < -_EPS).mean(axis=-1)                    # (B, K)

    posmask = evals > 0.0
    pos = np.where(posmask, evals, 0.0)                        # (B, K, d)
    s1 = pos.sum(axis=-1)
    s2 = (pos**2).sum(axis=-1)
    effrank = s1**2 / (s2 + _EPS)                              # (B, K)
    # log only over positive eigenvalues; feed a safe (>0) arg so np.where never logs a negative
    safe = np.where(posmask, evals, 1.0)
    logdet = np.where(posmask, np.log(safe + _EPS), 0.0).sum(axis=-1)

    best = np.asarray(traj.terminal_energy, dtype=float).argmin(axis=1)     # (B,)

    cols, names = [], []
    specs = ((lmin, "lmin"), (negfrac, "negfrac"), (effrank, "effrank"), (logdet, "logdet"))
    for arr, base in specs:
        mean_v, best_v = _mean_and_best(arr, best)
        cols += [mean_v, best_v]
        names += [f"spectrum/{base}_mean", f"spectrum/{base}_best"]

    return np.stack(cols, axis=1), names
