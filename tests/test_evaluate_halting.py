"""End-to-end adaptive-halting evaluation on the smoke config: complete, finite, JSON-serialisable
metrics with the F4 tau-sweep. Marked slow (it trains). Offline + CPU-only.
"""

import json

import numpy as np
import pytest

from edc.config import load_config
from edc.eval.evaluate_halting import evaluate_halting
from edc.registry import build_task

pytestmark = pytest.mark.slow


def test_halting_pipeline_metrics():
    cfg = load_config("configs/smoke.toml")
    task = build_task(cfg.run.task)
    m = evaluate_halting(cfg, task)

    for key in ("tau_hat", "compute_used", "halting_risk", "risk_within_budget",
                "full_accuracy", "halted_accuracy", "tau_sweep", "risk_monotone_in_tau"):
        assert key in m

    s = m["tau_sweep"]
    n = len(s["taus"])
    assert len(s["compute_used"]) == n and len(s["accuracy"]) == n and len(s["disagreement"]) == n
    assert all(0.0 <= c <= 1.0 for c in s["compute_used"])
    assert 0.0 <= m["full_accuracy"] <= 1.0 and np.isfinite(m["halted_accuracy"])
    json.dumps(m)  # ledger-serialisable
