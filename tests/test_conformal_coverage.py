"""Split-conformal coverage on synthetic exchangeable scores.

The marginal guarantee: with calibration size n and level alpha, a fresh point's score is
<= threshold with probability >= 1 - alpha. We check the *average* empirical coverage over many
random splits sits at or above 1 - alpha (and not wildly above).
"""

import numpy as np

from edc.conformal.split_conformal import conformal_threshold, empirical_coverage


def test_average_coverage_meets_target():
    rng = np.random.default_rng(0)
    alpha = 0.1
    n_cal, n_test, trials = 500, 2000, 200

    covers = []
    for _ in range(trials):
        scores = rng.standard_normal(n_cal + n_test)  # exchangeable
        cal, test = scores[:n_cal], scores[n_cal:]
        thr = conformal_threshold(cal, alpha)
        covers.append(empirical_coverage(test, thr))

    mean_cov = float(np.mean(covers))
    assert mean_cov >= 1 - alpha - 0.01     # guarantee (small MC slack)
    assert mean_cov <= 1 - alpha + 0.05     # not absurdly conservative


def test_threshold_admits_all_when_alpha_too_small_for_n():
    # With n=5 and alpha=0.01, ceil((n+1)(1-alpha)) > n -> cannot certify -> admit all.
    thr = conformal_threshold(np.arange(5.0), alpha=0.01)
    assert thr == float("inf")
