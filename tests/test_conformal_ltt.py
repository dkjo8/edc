"""Learn-then-Test: Hoeffding-Bentkus p-value validity + LTT selective-risk coverage guarantee.

The two properties that matter: (1) HB is a valid super-uniform p-value for ``H: R > alpha``, so
``P(p <= delta) <= delta`` when ``R == alpha``; (2) the calibrated threshold controls selective
risk at ``1 - delta`` confidence over fresh data. Offline + CPU-only.
"""

import numpy as np

from edc.conformal.ltt import calibrate, hoeffding_bentkus_pvalue, selective_risk


def test_hb_edges():
    assert hoeffding_bentkus_pvalue(0.3, 0.2, 200) == 1.0     # risk >= alpha: no evidence
    assert hoeffding_bentkus_pvalue(0.2, 0.2, 200) == 1.0     # boundary
    assert hoeffding_bentkus_pvalue(0.0, 0.2, 0) == 1.0       # empty sample certifies nothing
    p0 = hoeffding_bentkus_pvalue(0.0, 0.2, 5)               # finite, no nan at risk_hat=0
    assert np.isfinite(p0) and p0 == min(0.8 ** 5, np.e * 0.8 ** 5)


def test_hb_monotone_in_risk_hat():
    # p-value INCREASES as risk_hat rises toward alpha (evidence against H: R>alpha weakens).
    ps = [hoeffding_bentkus_pvalue(r, 0.3, 500) for r in np.linspace(0.0, 0.29, 30)]
    assert all(a <= b + 1e-12 for a, b in zip(ps[:-1], ps[1:], strict=True))  # non-decreasing


def test_hb_pvalue_is_valid_superuniform():
    # With true risk == alpha, the p-value must satisfy P(p <= delta) <= delta (validity).
    alpha, delta, n, trials = 0.2, 0.1, 200, 4000
    rng = np.random.default_rng(0)
    x = rng.binomial(n, alpha, size=trials)              # X ~ Bin(n, alpha)
    ps = np.array([hoeffding_bentkus_pvalue(xi / n, alpha, n) for xi in x])
    assert (ps <= delta).mean() <= delta + 0.02          # small Monte-Carlo slack


def test_selective_risk_empty_answered():
    r, n = selective_risk([0.5, 0.6], [True, True], lam=0.1)  # nothing answered
    assert n == 0 and np.isnan(r)


def _synth(rng, n, base_err):
    correct = rng.random(n) > base_err
    scores = np.where(correct, rng.beta(2, 5, n), rng.beta(5, 2, n))  # low score when correct
    return scores, correct


def test_ltt_coverage_guarantee():
    # Over many calib/test draws, the empirical P(R_sel(lambda_hat) <= alpha) must be >= 1 - delta.
    alpha, delta, trials = 0.1, 0.05, 250
    violations = 0
    coverages = []
    for t in range(trials):
        rng = np.random.default_rng(2000 + t)
        sc, cc = _synth(rng, 800, base_err=0.15)
        st, ct = _synth(rng, 4000, base_err=0.15)
        lam = calibrate(sc, cc, alpha, delta)["lambda_hat"]
        if lam is None:
            coverages.append(0.0)
            continue
        answered = st <= lam
        coverages.append(answered.mean())
        if answered.any() and (~ct[answered]).mean() > alpha:
            violations += 1
    assert violations / trials <= delta                   # distribution-free guarantee holds
    assert np.mean(coverages) > 0.5                        # and it is not vacuous (answers a lot)


def test_ltt_abstains_when_base_error_exceeds_alpha():
    # base error 25% > alpha 10%: must abstain on hard inputs -> coverage < 1, risk pulled down.
    rng = np.random.default_rng(7)
    sc, cc = _synth(rng, 3000, base_err=0.25)
    lam = calibrate(sc, cc, alpha=0.1, delta=0.05)["lambda_hat"]
    answered = sc <= lam
    assert 0.0 < answered.mean() < 1.0
    assert (~cc[answered]).mean() <= 0.15                 # selective risk near/below target
