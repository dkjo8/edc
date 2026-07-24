"""Split-conformal calibration (the distribution-free backbone).

Given exchangeable nonconformity scores on a held-out calibration set, the level-``alpha``
threshold is the ``ceil((n+1)(1-alpha))/n`` empirical quantile. Guarantee (marginal, under
exchangeability): a fresh point's score falls at or below the threshold with probability at
least ``1 - alpha``.

This is the coverage-only baseline. The *selective-error* guarantee (error rate among
answered inputs) needs Learn-then-Test (``edc.conformal.ltt``); the *halting* guarantee needs
Conformal Risk Control (``edc.conformal.crc``). Kept in NumPy so it never pulls in JAX.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Level-``alpha`` split-conformal threshold with the finite-sample correction."""
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = s.shape[0]
    if n == 0:
        raise ValueError("need at least one calibration score")
    # rank of the (1-alpha) quantile with the (n+1) correction; clamp to the top.
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k >= n:
        return float("inf")  # cannot certify at this alpha with n points -> admit all
    return float(s[k - 1])


def empirical_coverage(test_scores: np.ndarray, threshold: float) -> float:
    """Fraction of test points whose score is <= threshold."""
    return float(np.mean(np.asarray(test_scores, dtype=float) <= threshold))
