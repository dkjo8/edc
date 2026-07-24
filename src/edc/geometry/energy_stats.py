"""Terminal-energy statistics across restarts. [Phase 2]

Mean, min, and spread of the terminal energy over the K restarts. These reproduce the
**scalar-energy / best-of-N** signal that Energy-Based Transformers use — here they are just
features, and the falsification test is whether the geometry features beat them (ΔAURC).
"""

from __future__ import annotations


def energy_features(traj):
    raise NotImplementedError("Phase 2: mean/min/std of terminal energy across restarts.")
