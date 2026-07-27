"""Selective-prediction metrics: exact AURC values, risk-coverage shape, and the paired-bootstrap
ΔAURC falsification test (directionality + determinism).

Offline + CPU-only. NumPy-only maths; the only reason JAX loads is the seeding import.
"""

import numpy as np
import pytest

from edc.eval.metrics import (
    aurc,
    paired_bootstrap_delta_aurc,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)

T, F = True, False


def test_risk_coverage_exact():
    # scores ascending == confidence order; errors are the two least-confident points.
    cov, risk = risk_coverage_curve([0.1, 0.2, 0.3, 0.4], [T, T, F, F])
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0.0, 0.0, 1 / 3, 0.5])


def test_risk_coverage_full_risk_is_base_error():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(500)
    correct = rng.random(500) < 0.7
    _, risk = risk_coverage_curve(scores, correct)
    assert np.isclose(risk[-1], np.mean(~correct))  # full coverage == base error rate


def test_aurc_exact_value():
    assert aurc([0.1, 0.2, 0.3, 0.4], [T, T, F, F]) == pytest.approx(0.2083333, abs=1e-6)


def test_aurc_ranking_bounds():
    rng = np.random.default_rng(1)
    n = 400
    correct = rng.random(n) < 0.6
    error = (~correct).astype(float)
    perfect = error + rng.random(n) * 1e-6           # errors ranked LEAST confident (high score)
    worst = -perfect                                 # errors ranked MOST confident
    random_scores = rng.standard_normal(n)
    # perfect ranking pushes errors to the tail -> every prefix risk <= base error -> AURC minimal
    assert aurc(perfect, correct) < aurc(random_scores, correct) < aurc(worst, correct)
    assert aurc(perfect, correct) <= error.mean() and aurc(worst, correct) >= error.mean()
    # constant scores => all ties => tie-robust AURC == base error rate exactly
    assert aurc(np.ones(n), correct) == pytest.approx(error.mean())


def test_selective_accuracy_at_coverage():
    # answer the 2 most confident of 4 -> both correct -> selective acc 1.0
    assert selective_accuracy_at_coverage([0.1, 0.2, 0.3, 0.4], [T, T, F, F], 0.5) == 1.0
    assert selective_accuracy_at_coverage([0.1, 0.2, 0.3, 0.4], [T, T, F, F], 1.0) == 0.5


def test_delta_aurc_deterministic_via_seeding():
    rng = np.random.default_rng(2)
    n = 300
    correct = rng.random(n) < 0.6
    a, b = rng.random(n), rng.random(n)
    r1 = paired_bootstrap_delta_aurc(a, b, correct, n_boot=200, seed=5)
    r2 = paired_bootstrap_delta_aurc(a, b, correct, n_boot=200, seed=5)
    assert r1 == r2
    assert paired_bootstrap_delta_aurc(a, b, correct, n_boot=200, seed=6) != r1


def test_delta_aurc_directionality():
    rng = np.random.default_rng(3)
    n = 600
    correct = rng.random(n) < 0.6
    # b (geometry) ranks well; a (energy) is noise. delta = AURC(a) - AURC(b) > 0 => b wins.
    good = np.where(correct, rng.random(n) * 0.4, 0.6 + rng.random(n) * 0.4)
    noise = rng.random(n)
    delta, lo, hi = paired_bootstrap_delta_aurc(noise, good, correct, n_boot=1000, seed=0)
    assert delta > 0 and lo > 0                       # geometry strictly wins, CI excludes 0

    # identical scorers => delta 0 and CI brackets 0 (a genuine tie / null result)
    d0, lo0, hi0 = paired_bootstrap_delta_aurc(noise, noise, correct, n_boot=500, seed=0)
    assert d0 == 0.0 and lo0 <= 0.0 <= hi0
