"""Softmax-confidence baselines: MSP / entropy exact values, temperature scaling reduces NLL, and
degenerate-fold fallbacks. Offline + CPU-only, NumPy/scipy.
"""

import numpy as np

from edc.eval import baselines as B


def test_msp_and_entropy_exact():
    peaked = np.array([[10.0, 0.0, 0.0, 0.0]])
    uniform = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert B.msp_score(peaked)[0] < 1e-3                    # confident -> low nonconformity
    assert np.isclose(B.msp_score(uniform)[0], 0.75)       # 1 - 1/4
    assert B.entropy_score(peaked)[0] < 1e-2               # peaked -> ~0 entropy
    assert np.isclose(B.entropy_score(uniform)[0], np.log(4), atol=1e-6)   # uniform -> ln C


def test_softmax_normalises():
    p = B.softmax(np.random.default_rng(0).standard_normal((5, 7)))
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()


def test_fit_temperature_reduces_nll():
    rng = np.random.default_rng(0)
    n, c = 800, 5
    y = rng.integers(0, c, n)
    onehot = np.zeros((n, c))
    onehot[np.arange(n), y] = 1.0
    logits = (onehot + rng.standard_normal((n, c)) * 0.5) * 6.0   # mis-scaled (overconfident)
    T = B.fit_temperature(logits, y)
    assert 0.05 <= T <= 20.0
    assert B._nll(T, logits, y) <= B._nll(1.0, logits, y) + 1e-9  # temperature never hurts NLL


def test_temp_msp_and_single_class_fallback():
    logits = np.array([[3.0, 1.0, 0.0]])
    # higher temperature flattens the softmax -> higher nonconformity (less confident)
    assert B.temp_msp_score(logits, 5.0)[0] > B.temp_msp_score(logits, 0.5)[0]
    # a single-class fit fold cannot calibrate -> T = 1.0 fallback, scores still finite
    T = B.fit_temperature(logits.repeat(10, 0), np.zeros(10, int))
    assert T == 1.0
    assert np.all(np.isfinite(B.temp_msp_score(logits, T)))
