"""``evaluate(include_ood=False)`` skips the OOD fold but still returns the falsification + LTT
blocks. Marked slow (it trains a tiny reasoner). Offline + CPU-only.
"""

import numpy as np
import pytest

from edc.config import load_config
from edc.eval.evaluate import evaluate
from edc.registry import build_task

pytestmark = pytest.mark.slow


def test_include_ood_flag_toggles_ood_metrics():
    cfg = load_config("configs/smoke.toml")
    task = build_task(cfg.run.task)
    m = evaluate(cfg, task, include_ood=False)

    assert "accuracy_ood" not in m and "aurc_ood" not in m   # OOD fold skipped
    # ID falsification + guarantee still present and finite
    assert set(m) >= {"aurc", "delta_aurc_vs_best_energy", "geometry_wins", "ltt", "k_restarts"}
    d, lo, hi = m["delta_aurc_vs_best_energy"]
    assert lo <= d <= hi
    assert np.isfinite(m["accuracy_id"]) and "geometry" in m["aurc"]

    # Phase 4e softmax-confidence baselines + the vs-best-baseline falsification
    assert {"msp", "temp_msp", "entropy"} <= set(m["aurc"])
    assert m["best_baseline"] in m["aurc"]
    db, lob, hib = m["delta_aurc_vs_best_baseline"]
    assert lob <= db <= hib
    assert isinstance(m["geometry_wins_vs_baseline"], bool) and np.isfinite(m["temperature"])

    # Phase 4f feature-group leave-one-out ablation
    fa = m["feature_ablation"]
    assert {"full", "drop_basin", "drop_energy", "drop_curv", "drop_dynamics"} <= set(fa)
    assert all(0.0 <= v <= 1.0 for v in fa.values())

    # Phase 4j complementarity: learned-softmax vs geometry+softmax mappers
    assert {"softmax_learned", "geom_softmax"} <= set(m["aurc"])
    da, loa, hia = m["delta_aurc_geom_adds"]
    assert loa <= da <= hia
    assert isinstance(m["geometry_adds_over_softmax"], bool)
