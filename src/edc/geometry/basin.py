"""Basin-agreement features across restarts. [Phase 2]

Do independent restarts land in the same answer / same latent basin? Plurality fraction
``rho_basin``, decoded-answer entropy, and latent-cluster separation. Hypothesis: high
agreement => reliable.

Plain NumPy (invariant 1), fixed-shape and vectorized over the K axis.
"""

from __future__ import annotations

import numpy as np

from edc.inference.trajectory import TrajectoryRecord


def basin_features(traj: TrajectoryRecord) -> tuple[np.ndarray, list[str]]:
    """``(B, 3)`` basin-agreement features plus names.

    * ``basin/rho``        — plurality fraction of decoded answers across restarts (1 => all
      restarts agree).
    * ``basin/entropy``    — Shannon entropy (nats) of the decoded-answer distribution
      (0 => unanimous, higher => scattered).
    * ``basin/dispersion`` — mean pairwise Euclidean distance between restart endpoints
      ``z*`` (small => restarts converge to one basin).
    """
    pred = np.asarray(traj.pred)                                # (B, K)
    z = np.asarray(traj.z_star, dtype=float)                   # (B, K, d)
    B, K = pred.shape
    n_classes = int(np.asarray(traj.logits).shape[-1])

    # Per-input class counts across restarts -> (B, C).
    counts = np.zeros((B, n_classes), dtype=float)
    np.add.at(counts, (np.arange(B)[:, None], pred), 1.0)

    rho = counts.max(axis=1) / K
    probs = counts / K
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -(probs * logp).sum(axis=1)

    # Mean pairwise distance between the K endpoints, per input.
    diff = z[:, :, None, :] - z[:, None, :, :]                  # (B, K, K, d)
    dist = np.linalg.norm(diff, axis=-1)                       # (B, K, K)
    if K > 1:
        dispersion = dist.sum(axis=(1, 2)) / (K * (K - 1))     # off-diagonal mean
    else:
        dispersion = np.zeros(B)

    feats = np.stack([rho, entropy, dispersion], axis=1)
    return feats, ["basin/rho", "basin/entropy", "basin/dispersion"]
