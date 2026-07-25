"""Nonconformity mapper: learned score is monotone in a planted signal, standardisation travels
with the pickled object, the rho-basin fallback is exact, and a single-class fit fold degrades
gracefully. Offline + CPU-only.
"""

import pickle

import numpy as np

from edc.conformal.nonconformity import fit_mapper, rho_basin_score, score


def test_score_monotone_in_planted_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 3))
    correct = (x[:, 0] + rng.standard_normal(400) * 0.3) > 0   # feature 0 predicts correctness
    mapper = fit_mapper(x, correct)
    s = score(mapper, x)
    # higher feature 0 -> more likely correct -> higher p_hat -> LOWER nonconformity score
    assert np.corrcoef(s, x[:, 0])[0, 1] < -0.5
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_standardisation_travels_through_pickle():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 4)) * 100.0 + 5.0            # off-scale features
    correct = x[:, 1] > x[:, 1].mean()
    mapper = fit_mapper(x, correct)
    s = score(mapper, x)
    reloaded = pickle.loads(pickle.dumps(mapper))
    assert np.allclose(s, score(reloaded, x))                 # scaler stats persist with the model


def test_rho_basin_score_exact():
    assert np.allclose(rho_basin_score([1.0, 0.5, 0.25]), [0.0, 0.5, 0.75])


def test_single_class_fit_fold():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((50, 3))
    assert np.all(score(fit_mapper(x, np.zeros(50, bool)), x) == 1.0)   # all-wrong  -> s = 1
    assert np.all(score(fit_mapper(x, np.ones(50, bool)), x) == 0.0)    # all-correct -> s = 0
