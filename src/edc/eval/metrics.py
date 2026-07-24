"""Selective-prediction and calibration metrics. [Phase 3]

AURC (area under the risk-coverage curve, the primary metric), selective accuracy at fixed
coverage, ECE / adaptive ECE, and the paired-bootstrap ΔAURC(raw-energy - geometry) that backs
the falsification test. NumPy only.
"""

from __future__ import annotations

import numpy as np


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def risk_coverage_curve(scores, correct):  # F2 / AURC
    raise NotImplementedError("Phase 3: sort by score, sweep coverage, return (coverage, risk).")


def aurc(scores, correct) -> float:
    raise NotImplementedError("Phase 3: area under the risk-coverage curve.")


def paired_bootstrap_delta_aurc(scores_a, scores_b, correct, n_boot: int = 1000):
    raise NotImplementedError("Phase 3: bootstrap CI on AURC(a) - AURC(b); the falsification test.")
