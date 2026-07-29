"""Mode-connectivity features between restart basins. [Phase 4k, richer geometry]

Phase 2's ``basin/dispersion`` measures only the *Euclidean* distance between the K restart
endpoints — it cannot tell whether two endpoints sit in genuinely separate basins (a high energy
barrier between them) or in the same basin reached from different directions (no barrier). This
module walks the straight-line path between endpoints and reads the **energy barrier** off the
actual landscape:

    barrier(i, j) = max_t E(z(t)) - max(E_i, E_j),   z(t) = (1-t) z*_i + t z*_j.

To keep cost ~K energy paths per input (not K^2) we measure barriers from the **best** (min-energy)
restart to every other restart — the "how separated is the answer we would report from the
alternatives" question.

Plain NumPy assembler (invariant 1): the only JAX is behind ``curvature.batched_path_energy``.
"""

from __future__ import annotations

import numpy as np

from edc.geometry import curvature
from edc.inference.trajectory import TrajectoryRecord

_N_POINTS = 8
_CONNECTED_EPS = 1e-3


def connectivity_features(traj: TrajectoryRecord, fns, params) -> tuple[np.ndarray, list[str]]:
    """``(B, 3)`` mode-connectivity features plus names.

    * ``connect/barrier_mean``   — mean energy barrier from the best restart to the others.
    * ``connect/barrier_max``    — largest such barrier (most separated alternative basin).
    * ``connect/connected_frac`` — fraction of the other restarts reachable with barrier
      ``< eps`` (i.e. in the same basin as the reported answer).

    With ``K == 1`` there are no pairs; all three features are 0.
    """
    z = np.asarray(traj.z_star, dtype=float)                   # (B, K, d)
    B, K, d = z.shape
    names = ["connect/barrier_mean", "connect/barrier_max", "connect/connected_frac"]
    if K < 2:
        return np.zeros((B, 3)), names

    h_x = np.asarray(traj.h_x, dtype=float)                    # (B, dc)
    best = np.asarray(traj.terminal_energy, dtype=float).argmin(axis=1)     # (B,)

    # Build B*(K-1) endpoint pairs: (best restart) -> (each other restart).
    rows = np.arange(B)
    z_best = z[rows, best]                                     # (B, d)
    others = np.stack([[k for k in range(K) if k != b_best] for b_best in best])  # (B, K-1)
    z_other = z[rows[:, None], others]                        # (B, K-1, d)

    n_pairs = B * (K - 1)
    z_a = np.repeat(z_best, K - 1, axis=0)                    # (n_pairs, d)
    z_b = z_other.reshape(n_pairs, d)                         # (n_pairs, d)
    contexts = np.repeat(h_x, K - 1, axis=0)                  # (n_pairs, dc)

    path = np.asarray(
        curvature.batched_path_energy(fns.energy, params, contexts, z_a, z_b, n_points=_N_POINTS),
        dtype=float,
    )                                                         # (n_pairs, M)
    endpoints_max = np.maximum(path[:, 0], path[:, -1])
    barrier = (path.max(axis=1) - endpoints_max).reshape(B, K - 1)          # (B, K-1)
    barrier = np.maximum(barrier, 0.0)                        # a barrier is non-negative by defn

    feats = np.stack(
        [barrier.mean(axis=1), barrier.max(axis=1), (barrier < _CONNECTED_EPS).mean(axis=1)],
        axis=1,
    )
    return feats, names
