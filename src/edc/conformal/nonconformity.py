"""Nonconformity scores from geometry features. [Phase 3]

Fits a small mapper (logistic regression) from geometry features to ``p_hat(correct)`` on a fold
**disjoint** from the calibration fold (invariant 7), then scores ``s(x) = 1 - p_hat(correct)``
(low ``s`` = confident). A hand-defined ``1 - rho_basin`` variant is kept so conformal validity
does not hinge on the learned mapper.

Plain NumPy / scikit-learn (invariant 1): no JAX. The standardisation statistics travel *inside*
the returned sklearn ``Pipeline`` so ``score`` is a pure function of the fitted object.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class _ConstantMapper:
    """Degenerate mapper for a single-class fit fold: ``p_hat(correct)`` is a constant.

    Exposes the minimal sklearn surface (``classes_`` + ``predict_proba``) that ``score`` uses, so
    the all-correct / all-wrong fold produces a consistent constant nonconformity score instead of
    crashing ``LogisticRegression`` (which needs >= 2 classes).
    """

    def __init__(self, cls: int) -> None:
        self.classes_ = np.array([cls])

    def predict_proba(self, x):
        return np.ones((np.asarray(x).shape[0], 1))


def fit_mapper(features_train, correct_train):
    """Fit ``features -> p_hat(correct)`` on the FIT fold (disjoint from calibration, invariant 7).

    Returns a standardise-then-logistic ``Pipeline``; the scaler's mean/scale are stored on the
    object so ``score`` needs nothing else. A single-class fit fold (all-correct or all-wrong)
    yields a ``_ConstantMapper`` so ``score`` stays well-defined.
    """
    x = np.asarray(features_train, dtype=float)
    y = np.asarray(correct_train).astype(int)
    classes = np.unique(y)
    if classes.shape[0] < 2:
        return _ConstantMapper(int(classes[0]))
    mapper = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    mapper.fit(x, y)
    return mapper


def score(mapper: Pipeline, features) -> np.ndarray:
    """Nonconformity score ``s = 1 - p_hat(correct)`` for each row of ``features``.

    Looks up the positive-class (``correct == 1``) column via ``classes_`` rather than assuming
    column 1; if the fit fold saw only one class, returns a constant score consistent with it.
    """
    x = np.asarray(features, dtype=float)
    classes = mapper.classes_
    proba = mapper.predict_proba(x)
    if 1 in classes:
        p_correct = proba[:, list(classes).index(1)]
    else:
        # Fit fold had no correct examples -> model always predicts incorrect -> p_correct = 0.
        p_correct = np.zeros(x.shape[0])
    return 1.0 - p_correct


def rho_basin_score(rho) -> np.ndarray:
    """Hand-defined fallback score ``s = 1 - rho_basin`` (no fit).

    ``rho`` is the plurality fraction of decoded answers across restarts
    (``basin_features(traj)[0][:, 0]``). High agreement => confident => low score. Kept so
    distribution-free validity never depends on the learned mapper (invariant 7).
    """
    return 1.0 - np.asarray(rho, dtype=float)
