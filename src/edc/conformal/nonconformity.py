"""Nonconformity scores from geometry features. [Phase 3]

Fits a small monotone mapper (logistic regression / gradient-boosted trees) from geometry
features to ``p_hat(correct)`` on a fold **disjoint** from the calibration fold (invariant 7),
then scores ``s(x) = 1 - p_hat(correct)``. A hand-defined ``1 - rho_basin`` variant is kept so
conformal validity does not hinge on the learned mapper.
"""

from __future__ import annotations


def fit_mapper(features_train, correct_train):
    raise NotImplementedError("Phase 3: fit logistic/GBT features -> p(correct) on the fit fold.")


def score(mapper, features):
    raise NotImplementedError("Phase 3: s(x) = 1 - p_hat(correct).")
