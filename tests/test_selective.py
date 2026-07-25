"""Selective-prediction wrapper: exact abstain behaviour (boundary answers) and the coverage
identity linking it to the conformal layer. Offline + CPU-only.
"""

import numpy as np

from edc.conformal.selective import ABSTAIN, selective_predict
from edc.conformal.split_conformal import empirical_coverage


def test_selective_predict_exact_with_boundary():
    ans = selective_predict([3, 1, 4, 1], [0.1, 0.9, 0.5, 0.5], threshold=0.5)
    assert list(ans) == [3, ABSTAIN, 4, 1]                 # s == threshold answers


def test_selective_predict_extremes():
    pred = np.array([7, 8, 9])
    assert list(selective_predict(pred, [0.1, 0.2, 0.3], float("inf"))) == [7, 8, 9]  # admit all
    assert list(selective_predict(pred, [0.1, 0.2, 0.3], float("-inf"))) == [ABSTAIN] * 3


def test_coverage_matches_empirical_coverage():
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    pred = rng.integers(0, 5, size=500)
    thr = 0.4
    answers = selective_predict(pred, scores, thr)
    assert np.isclose((answers != ABSTAIN).mean(), empirical_coverage(scores, thr))
