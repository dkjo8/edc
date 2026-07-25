"""Selective-prediction wrapper: answer or ABSTAIN. [Phase 3]

Wraps a trained reasoner + a calibrated threshold (from ``ltt``): answer iff ``s(x) <= lambda``,
else abstain and route to a human/verifier — the critical-systems behaviour. Reports selective
accuracy, coverage, and abstention rate.
"""

from __future__ import annotations

import numpy as np

ABSTAIN = -1


def selective_predict(pred, scores, threshold) -> np.ndarray:
    """Return ``pred`` where ``score <= threshold`` (answer), else ``ABSTAIN`` (-1).

    Boundary answers (``s == threshold``), consistent with ``split_conformal.empirical_coverage``
    and the ``s <= lambda`` convention used across the conformal layer. ``threshold`` may be
    ``inf`` (admit all) or ``-inf`` (abstain all).
    """
    pred = np.asarray(pred)
    scores = np.asarray(scores, dtype=float)
    if pred.shape[0] != scores.shape[0]:
        raise ValueError("pred and scores must have the same length")
    return np.where(scores <= threshold, pred, ABSTAIN)
