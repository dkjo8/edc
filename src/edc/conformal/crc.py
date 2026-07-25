"""Conformal Risk Control for adaptive halting. [Phase 3]

The halting risk ``L(lambda) = 1[early-stopped answer != full-budget answer]`` is bounded and
**monotone** in a single threshold ``lambda`` (relax lambda => stop earlier => weakly more
errors). CRC (Angelopoulos et al., arXiv:2208.02814) then picks ``lambda_hat`` so
``E[L(lambda_hat)] <= alpha`` in finite samples. Monotonicity is exactly why CRC — not LTT —
is the right tool here.

This module is the pure calibrator. The halting *policy* that produces ``loss_by_lambda`` from a
trajectory (``halting.adaptive``) and figure F4 are deferred (they need per-step decoded answers
on ``TrajectoryRecord``). NumPy only (invariant 1).
"""

from __future__ import annotations

import numpy as np


def _normalise(loss_by_lambda, lambdas):
    """Return ``(lambdas ascending, loss_matrix (n_cal, n_lambda) aligned to them)``.

    Accepts a ``dict {lambda: per-point loss vector}`` or a 2D array with an explicit ``lambdas``.
    Lambdas are sorted ascending; by convention a larger lambda halts earlier => saves more
    compute => weakly higher loss.
    """
    if isinstance(loss_by_lambda, dict):
        lams = np.array(sorted(loss_by_lambda), dtype=float)
        mat = np.stack([np.asarray(loss_by_lambda[k], dtype=float) for k in lams], axis=1)
        return lams, mat
    mat = np.asarray(loss_by_lambda, dtype=float)
    if mat.ndim != 2:
        raise ValueError("loss_by_lambda must be a dict or a (n_cal, n_lambda) 2D array")
    lams = np.arange(mat.shape[1], dtype=float) if lambdas is None else np.asarray(lambdas, float)
    order = np.argsort(lams)
    return lams[order], mat[:, order]


def calibrate(loss_by_lambda, alpha: float, lambdas=None):
    """Pick the most compute-saving ``lambda_hat`` whose CRC bound holds, or ``None`` if infeasible.

    Finite-sample CRC bound with bounded loss ``B = 1`` (mirrors the ``(n+1)`` correction in
    ``split_conformal``)::

        (n / (n + 1)) * R_hat(lambda) + 1 / (n + 1) <= alpha

    where ``R_hat`` is the mean loss over the ``n`` calibration points. Because loss is monotone
    non-decreasing in ``lambda`` (more compute saved => weakly more errors), the admissible set is
    a prefix in ascending-lambda order; we scan that order and return the **largest admissible**
    lambda (maximal compute saving while safe). Returns ``None`` when even the most conservative
    lambda violates the bound — including the structurally infeasible regime ``alpha < 1/(n+1)``,
    where no risk (not even 0) can satisfy it.
    """
    lams, mat = _normalise(loss_by_lambda, lambdas)
    n = mat.shape[0]
    if n == 0:
        raise ValueError("need at least one calibration point")
    rhs = (alpha * (n + 1) - 1.0) / n  # R_hat must be <= this
    r_hat = mat.mean(axis=0)

    lambda_hat = None
    for lam, r in zip(lams, r_hat, strict=True):
        if r <= rhs:
            lambda_hat = float(lam)  # extend the admissible prefix
        else:
            break  # monotone: once it fails, larger lambdas only get worse
    return lambda_hat
