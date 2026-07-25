"""Selective-prediction and calibration metrics. [Phase 3]

AURC (area under the risk-coverage curve, the primary metric), selective accuracy at fixed
coverage, ECE / adaptive ECE, and the paired-bootstrap ΔAURC(raw-energy - geometry) that backs
the falsification test. NumPy only.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from edc.seeding import numpy_rng


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def single_feature_auroc(feature: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of a single scalar ``feature`` separating correct from incorrect answers.

    ``P(feature[correct] > feature[incorrect])`` via the Mann-Whitney rank statistic (ties
    count as 0.5). A value near 0.5 means no separation; far from 0.5 (in either direction —
    the feature's sign is not assumed) means the feature carries correctness signal. This is a
    Phase-2 *diagnostic*; the Phase-3 selective-prediction metric is ``aurc`` (still stubbed).
    Returns 0.5 when either class is empty (undefined).
    """
    feature = np.asarray(feature, dtype=float)
    correct = np.asarray(correct).astype(bool)
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(feature)  # average ranks, ties shared
    return float((ranks[correct].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _errors_by_confidence(scores: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Per-point error (``~correct``) reordered most-confident first, tie-robust.

    Convention: the nonconformity ``score`` is LOW when confident, so an ascending sort puts the
    answers we would keep first. **Tied scores share their group-mean error** — the expected risk
    under random tie-breaking — so the curve (and AURC) does not depend on incidental input order
    within a tie group. This matters for scores with few distinct values (e.g. ``rho_basin`` over
    K restarts) so the geometry-vs-energy comparison stays fair.
    """
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct).astype(bool)
    if scores.shape[0] != correct.shape[0]:
        raise ValueError("scores and correct must have the same length")
    if scores.shape[0] == 0:
        raise ValueError("need at least one point")
    order = np.argsort(scores, kind="stable")
    err = (~correct[order]).astype(float)
    # Replace each point's error by its tie-group mean (groups are contiguous after the sort).
    _, inv = np.unique(scores[order], return_inverse=True)
    inv = inv.ravel()
    group_mean = np.bincount(inv, weights=err) / np.bincount(inv)
    return group_mean[inv]


def risk_coverage_curve(scores, correct):  # F2 / AURC
    """Return ``(coverage, risk)``, each ``(n,)``, sweeping coverage from the most-confident point.

    At coverage ``c = (i+1)/n`` we answer the ``i+1`` lowest-score inputs; ``risk`` is the error
    rate among them (``cumsum(error)/(i+1)``). ``risk[-1]`` is the base error rate (full coverage).
    Ties in ``scores`` split across the prefix boundary — fine for the scalar AURC summary and for
    a diagnostic curve (conformal thresholds admit whole tie-groups anyway).
    """
    e = _errors_by_confidence(scores, correct)
    n = e.shape[0]
    coverage = np.arange(1, n + 1, dtype=float) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    return coverage, risk


def aurc(scores, correct) -> float:
    """Area under the risk-coverage curve = mean selective risk over the uniform coverage grid.

    Lower is better: a ranker that pushes all errors to the least-confident tail minimises every
    prefix risk. Equals the base error rate when scores carry no ranking signal (all ties).
    """
    e = _errors_by_confidence(scores, correct)
    return float(np.mean(e.cumsum() / np.arange(1, e.shape[0] + 1)))


def selective_accuracy_at_coverage(scores, correct, coverage: float) -> float:
    """Selective accuracy (``1 - risk``) when answering the most-confident ``coverage`` fraction."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    e = _errors_by_confidence(scores, correct)
    k = max(1, int(round(coverage * e.shape[0])))
    return float(1.0 - e[:k].mean())


def ece(prob_correct, correct, n_bins: int = 10) -> float:
    """Expected calibration error of a ``p_hat(correct)`` estimate, equal-width binning.

    ``sum_b (n_b/n) * |acc_b - conf_b|`` — the standard reliability-diagram summary. Used as a
    reported calibration number alongside the distribution-free guarantees.
    """
    p = np.asarray(prob_correct, dtype=float)
    y = np.asarray(correct).astype(float)
    if p.shape[0] == 0:
        raise ValueError("need at least one point")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def paired_bootstrap_delta_aurc(scores_a, scores_b, correct, n_boot: int = 1000, seed: int = 0):
    """Paired-bootstrap CI on ``ΔAURC = AURC(a) - AURC(b)`` — the falsification test (invariant 8).

    ``a`` = raw terminal energy, ``b`` = geometry. Each bootstrap iterate resamples one index
    vector with replacement and applies it to *both* scorers and ``correct`` (the pairing cancels
    shared test-set variance). Returns ``(delta, lo, hi)`` with a 95% percentile CI. **Geometry
    wins iff ``delta > 0`` and ``lo > 0``** (its AURC is lower); a CI containing 0 falsifies the
    core claim. Randomness flows only through ``edc.seeding`` (invariant 4).
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    c = np.asarray(correct).astype(bool)
    n = c.shape[0]
    if not (sa.shape[0] == sb.shape[0] == n):
        raise ValueError("scores_a, scores_b, correct must have equal length")

    delta = aurc(sa, c) - aurc(sb, c)
    rng = numpy_rng(seed, 101)  # 101 = the ΔAURC bootstrap substream
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = aurc(sa[idx], c[idx]) - aurc(sb[idx], c[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return float(delta), float(lo), float(hi)
