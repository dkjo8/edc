"""F5/F6 metric blocks from the evaluate pipeline: OOD selective-risk validity (guarantee applied
under shift) and per-feature diagnostics. Marked slow (it trains). Offline + CPU-only.
"""

import json

import numpy as np
import pytest

from edc.config import load_config
from edc.eval.evaluate import _ALPHA_GRID, evaluate
from edc.registry import build_task

pytestmark = pytest.mark.slow


def test_ood_validity_and_feature_diagnostics_present():
    cfg = load_config("configs/smoke.toml")
    task = build_task(cfg.run.task)
    m = evaluate(cfg, task, include_ood=True)

    # F6: ood_validity mirrors the alpha grid with ID + OOD achieved risk, and an ood_ltt block.
    ov = m["ood_validity"]
    n = len(_ALPHA_GRID)
    for key in ("target", "id_risk", "ood_risk", "id_coverage", "ood_coverage"):
        assert len(ov[key]) == n
    assert set(m["ood_ltt"]) >= {"alpha", "lambda_hat", "selective_risk", "coverage",
                                 "risk_within_budget"}

    # F5: 14 named per-feature AUROCs + histograms whose counts sum to the class sizes.
    fd = m["feature_diagnostics"]
    assert len(fd["auroc"]) == 14 and len(fd["names"]) == 14
    n_correct = int(np.round(m["accuracy_id"] * m["n_test"]))
    h = fd["hist"][fd["names"][0]]
    assert len(h["edges"]) == 21                                  # 20 bins
    assert sum(h["correct_counts"]) == n_correct
    assert sum(h["correct_counts"]) + sum(h["incorrect_counts"]) == m["n_test"]

    json.dumps(m)  # ledger-serialisable


def test_include_ood_false_omits_f6_blocks():
    cfg = load_config("configs/smoke.toml")
    task = build_task(cfg.run.task)
    m = evaluate(cfg, task, include_ood=False)
    assert "ood_validity" not in m and "ood_ltt" not in m
    assert "feature_diagnostics" in m                             # F5 does not need the OOD fold
