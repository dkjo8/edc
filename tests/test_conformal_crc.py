"""Conformal Risk Control calibrator: exact finite-sample boundary, infeasible-alpha guard, and
the monotone-halting risk guarantee ``E[L(lambda_hat)] <= alpha``. Offline + CPU-only.
"""

import numpy as np

from edc.conformal.crc import calibrate


def test_crc_boundary_pick():
    # n=9, alpha=0.2 -> RHS = ((n+1)*alpha - 1)/n = (2 - 1)/9 = 0.1111.
    # Three lambdas with R_hat = {0, 1/9=0.111, 1}. Admissible prefix is {0, 0.111}; pick the
    # most compute-saving (largest index) admissible -> lambda index 1.
    n = 9
    loss = np.zeros((n, 3))
    loss[0, 1] = 1.0          # column 1: one error -> R_hat = 1/9 = 0.111 <= 0.111 (admissible)
    loss[:, 2] = 1.0          # column 2: R_hat = 1.0 (fails)
    assert calibrate(loss, alpha=0.2, lambdas=[0.0, 0.5, 1.0]) == 0.5


def test_crc_infeasible_alpha_returns_none():
    # n=3, alpha=0.2 -> (n+1)*alpha - 1 = -0.2 < 0: no R_hat (not even 0) can satisfy the bound.
    assert calibrate(np.zeros((3, 2)), alpha=0.2) is None


def test_crc_dict_input_and_monotone_stop():
    # dict form; risks increase with lambda. RHS for n=20, alpha=0.2 = (21*0.2-1)/20 = 0.16.
    n = 20
    loss_by_lambda = {
        0.1: np.zeros(n),                                    # R_hat 0.00 -> admissible
        0.2: np.array([1] * 3 + [0] * (n - 3)),              # R_hat 0.15 -> admissible
        0.3: np.array([1] * 5 + [0] * (n - 5)),              # R_hat 0.25 -> fails, stop
    }
    assert calibrate(loss_by_lambda, alpha=0.2) == 0.2


def test_crc_risk_guarantee():
    # Monte-Carlo: E[L(lambda_hat)] <= alpha over draws. Loss at each lambda is Bernoulli with a
    # rate that increases with lambda; CRC must never pick a lambda whose true rate exceeds alpha.
    alpha = 0.1
    lambdas = np.array([0.0, 1.0, 2.0, 3.0])
    true_rates = np.array([0.02, 0.06, 0.12, 0.30])
    picked_true_rate = []
    for t in range(400):
        rng = np.random.default_rng(500 + t)
        loss = (rng.random((300, len(lambdas))) < true_rates).astype(float)
        lam = calibrate(loss, alpha, lambdas=lambdas)
        if lam is not None:
            picked_true_rate.append(true_rates[list(lambdas).index(lam)])
    # the finite-sample bound should keep the expected picked risk at or below alpha
    assert np.mean(picked_true_rate) <= alpha
