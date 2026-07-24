"""Selective-prediction wrapper: answer or ABSTAIN. [Phase 3]

Wraps a trained reasoner + a calibrated threshold (from ``ltt``): answer iff ``s(x) <= lambda``,
else abstain and route to a human/verifier — the critical-systems behaviour. Reports selective
accuracy, coverage, and abstention rate.
"""

from __future__ import annotations

ABSTAIN = -1


def selective_predict(pred, scores, threshold):
    raise NotImplementedError("Phase 3: return pred where score<=threshold else ABSTAIN.")
