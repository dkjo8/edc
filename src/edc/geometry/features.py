"""Assemble the per-input geometry feature vector from a TrajectoryRecord. [Phase 2]

Combines: basin agreement (``basin``), terminal-energy statistics (``energy_stats``),
curvature (``curvature``), and descent dynamics (``dynamics``) into the nonconformity
feature vector that Phase-3 conformal calibration consumes. This is the paper's contribution.

Plain NumPy assembler (invariant 1): the only JAX touched here is behind
``curvature.batched_curvature``, whose output is immediately pulled back to NumPy.
"""

from __future__ import annotations

import numpy as np

from edc.geometry import curvature
from edc.geometry.basin import basin_features
from edc.geometry.connectivity import connectivity_features
from edc.geometry.dynamics import dynamics_features
from edc.geometry.energy_stats import energy_features
from edc.geometry.spectrum import spectrum_features
from edc.inference.trajectory import TrajectoryRecord


def _curvature_features(traj, fns, params, key) -> tuple[np.ndarray, list[str]]:
    """``(B, 4)`` curvature summaries: mean over restarts and value at the min-energy restart."""
    z = np.asarray(traj.z_star, dtype=float)                   # (B, K, d)
    B, K, d = z.shape
    h_x = np.asarray(traj.h_x, dtype=float)                    # (B, dc)

    # Flatten to N = B*K particles exactly as inference.restarts lays them out (particle
    # (b, k) at index b*K + k), tiling the shared per-input context to match.
    contexts = np.repeat(h_x, K, axis=0)                       # (N, dc)
    zs = z.reshape(B * K, d)                                   # (N, d)

    lmax, trace = curvature.batched_curvature(fns.energy, params, contexts, zs, key)
    lmax = np.asarray(lmax, dtype=float).reshape(B, K)
    trace = np.asarray(trace, dtype=float).reshape(B, K)

    # "Best" restart = lowest terminal energy (the best-of-N answer we would report).
    best = np.asarray(traj.terminal_energy, dtype=float).argmin(axis=1)   # (B,)
    rows = np.arange(B)
    feats = np.stack(
        [lmax.mean(axis=1), lmax[rows, best], trace.mean(axis=1), trace[rows, best]],
        axis=1,
    )
    return feats, ["curv/lmax_mean", "curv/lmax_best", "curv/trace_mean", "curv/trace_best"]


def geometry_features(
    traj: TrajectoryRecord, fns, params, key, grad_tol: float = 1e-3, richer: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Return ``((B, n_features) array, feature-name list)``.

    Concatenates the four base feature groups (basin, energy stats, curvature, dynamics) in a
    fixed order — 14 features. ``key`` seeds the curvature HVP estimators; ``grad_tol`` sets the
    convergence threshold for the dynamics ``steps`` feature (pass ``cfg.inference.grad_tol``).

    ``richer=True`` (Phase 4k, opt-in) appends two richer groups **after** the base 14 — the full
    Hessian spectrum (``spectrum/*``) and mode-connectivity barriers (``connect/*``) — so the base
    column order/indices are unchanged and every prior 14-feature result is reproduced exactly.
    """
    b_feats, b_names = basin_features(traj)
    e_feats, e_names = energy_features(traj)
    c_feats, c_names = _curvature_features(traj, fns, params, key)
    d_feats, d_names = dynamics_features(traj, grad_tol)

    groups = [(b_feats, b_names), (e_feats, e_names), (c_feats, c_names), (d_feats, d_names)]
    if richer:
        groups.append(spectrum_features(traj, fns, params))
        groups.append(connectivity_features(traj, fns, params))

    feats = np.concatenate([g for g, _ in groups], axis=1)
    names = [n for _, ns in groups for n in ns]
    return feats, names
