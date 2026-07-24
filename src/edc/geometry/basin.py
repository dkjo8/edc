"""Basin-agreement features across restarts. [Phase 2]

Do independent restarts land in the same answer / same latent basin? Plurality fraction
``rho_basin``, decoded-answer entropy, and latent-cluster separation. Hypothesis: high
agreement => reliable. Must be computable under ``vmap`` over restarts.
"""

from __future__ import annotations


def basin_features(traj):
    raise NotImplementedError("Phase 2: rho_basin, answer entropy, latent cluster separation.")
