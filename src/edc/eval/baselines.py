"""Standard softmax-confidence nonconformity baselines. [Phase 4e]

Geometry must beat not only the EBT scalar energy (invariant 8) but also the obvious decoder-softmax
confidence signals. These are cheap, standard baselines computed from the mean decoder logits over
the K restarts (a logit-averaging ensemble read-off), conformalized identically to geometry:

* **MSP** — ``1 - max_c softmax``: the classic maximum-softmax-probability confidence.
* **temperature-scaled MSP** — MSP after a single scalar temperature fit on the (disjoint) fit fold.
* **predictive entropy** — entropy of the softmax (higher = less confident).

All return nonconformity scores (low = confident), so ``eval.metrics.aurc`` consumes them directly.
NumPy/scipy only (invariant 1); temperature fit is deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of ``(n, C)`` logits."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def msp_score(mean_logits: np.ndarray) -> np.ndarray:
    """``1 - max softmax`` — low when the predictive distribution is peaked (confident)."""
    return 1.0 - softmax(mean_logits).max(axis=-1)


def entropy_score(mean_logits: np.ndarray) -> np.ndarray:
    """Predictive entropy in nats — high when the distribution is flat (unconfident)."""
    p = softmax(mean_logits)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    return -(p * logp).sum(axis=-1)


def _nll(temp: float, logits: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of ``y`` under ``softmax(logits / temp)``."""
    p = softmax(logits / temp)
    n = logits.shape[0]
    return float(-np.mean(np.log(p[np.arange(n), y] + 1e-12)))


def fit_temperature(mean_logits_fit: np.ndarray, y_fit: np.ndarray) -> float:
    """Scalar temperature minimising NLL on the fit fold (invariant 7); ``1.0`` fallback.

    Standard temperature scaling (Guo et al. 2017), fit on a fold disjoint from calibration so the
    resulting nonconformity score stays exchangeable with the calibration set.
    """
    logits = np.asarray(mean_logits_fit, dtype=float)
    y = np.asarray(y_fit).astype(int)
    if logits.shape[0] == 0 or len(np.unique(y)) < 2:
        return 1.0
    res = minimize_scalar(_nll, bounds=(0.05, 20.0), args=(logits, y), method="bounded")
    return float(res.x) if res.success else 1.0


def temp_msp_score(mean_logits: np.ndarray, temp: float) -> np.ndarray:
    """MSP after temperature scaling: ``1 - max softmax(logits / temp)``."""
    return 1.0 - softmax(np.asarray(mean_logits, dtype=float) / temp).max(axis=-1)
