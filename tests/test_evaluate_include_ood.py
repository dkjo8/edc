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

    # base run carries the Phase-4k feature-set tag and no richer feature columns
    assert m["feature_set"] == "base"
    assert not any(n.startswith(("spectrum/", "connect/")) for n in m["feature_names"])

    # no ensemble by default (ensemble_size == 1)
    assert "delta_aurc_vs_best_ensemble" not in m
    assert not ({"ens_msp", "ens_entropy", "ens_disagreement"} & set(m["aurc"]))


def test_deep_ensemble_baseline_wires_in():
    from dataclasses import replace

    cfg = load_config("configs/smoke.toml")
    cfg = replace(cfg, eval=replace(cfg.eval, ensemble_size=2))
    task = build_task(cfg.run.task)
    m = evaluate(cfg, task, include_ood=False)

    # Phase 4m: ensemble scores get AURCs and a separate (not folded into baselines) comparison.
    assert {"ens_msp", "ens_entropy", "ens_disagreement"} <= set(m["aurc"])
    assert m["ensemble_size"] == 2
    assert m["best_ensemble"] in ("ens_msp", "ens_entropy", "ens_disagreement")
    de, loe, hie = m["delta_aurc_vs_best_ensemble"]
    assert loe <= de <= hie
    assert isinstance(m["geometry_beats_ensemble"], bool)
    assert 0.0 <= m["ensemble_accuracy"] <= 1.0
    # the same-compute baseline set is untouched (ensemble reported separately)
    assert m["best_baseline"] in ("energy_min", "energy_mean", "energy_std",
                                  "msp", "temp_msp", "entropy")


def test_deep_ensemble_is_deterministic():
    from dataclasses import replace

    cfg = load_config("configs/smoke.toml")
    cfg = replace(cfg, eval=replace(cfg.eval, ensemble_size=2))
    task = build_task(cfg.run.task)
    m1 = evaluate(cfg, task, include_ood=False)
    m2 = evaluate(cfg, task, include_ood=False)
    assert m1["aurc"]["ens_msp"] == m2["aurc"]["ens_msp"]           # same seed -> identical
    assert m1["delta_aurc_vs_best_ensemble"] == m2["delta_aurc_vs_best_ensemble"]


def test_richer_geometry_appends_groups_and_tags_feature_set():
    from dataclasses import replace

    cfg = load_config("configs/smoke.toml")
    cfg = replace(cfg, eval=replace(cfg.eval, richer_geometry=True))
    task = build_task(cfg.run.task)
    m = evaluate(cfg, task, include_ood=False)

    # Phase 4k: the richer groups flow into the feature names, AURC diagnostics, and ablation.
    assert m["feature_set"] == "richer"
    names = m["feature_names"]
    assert any(n.startswith("spectrum/") for n in names)
    assert any(n.startswith("connect/") for n in names)
    fa = m["feature_ablation"]
    assert {"drop_spectrum", "drop_connect", "spectrum_only", "connect_only"} <= set(fa)
    assert all(0.0 <= v <= 1.0 for v in fa.values())
    # the complementarity metric still computes on the richer feature set
    da, loa, hia = m["delta_aurc_geom_adds"]
    assert loa <= da <= hia
