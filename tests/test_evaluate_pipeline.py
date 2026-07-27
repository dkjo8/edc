"""End-to-end selective-prediction pipeline on the tiny smoke config.

Trains a small reasoner, runs the disjoint fit/calib/test folds, and checks the metrics dict is
complete, finite, JSON-serialisable, and keeps the raw-energy baseline wired in (invariant 8).
Marked ``slow`` (it trains) but still offline + CPU-only. Values are not asserted — the smoke
config saturates; correctness of the numbers is covered by the unit tests.
"""

import json

import numpy as np
import pytest

from edc.config import load_config
from edc.eval.evaluate import evaluate
from edc.registry import build_task
from edc.seeding import numpy_rng

pytestmark = pytest.mark.slow


def test_pipeline_returns_complete_finite_metrics():
    cfg = load_config("configs/smoke.toml")
    task = build_task(cfg.run.task)
    m = evaluate(cfg, task)

    # folds are the three disjoint sizes from the config
    assert m["n_fit"] == cfg.eval.n_eval
    assert m["n_calib"] == cfg.conformal.n_calib
    assert m["n_test"] == cfg.eval.n_eval

    # raw-energy baselines present alongside geometry (invariant 8) + the fallback
    for name in ("geometry", "rho_basin", "energy_min", "energy_mean", "energy_std"):
        assert name in m["aurc"]
        assert 0.0 <= m["aurc"][name] <= 1.0

    # falsification test wired: ΔAURC point estimate + ordered CI
    d, lo, hi = m["delta_aurc_vs_best_energy"]
    assert lo <= d <= hi
    assert isinstance(m["geometry_wins"], bool)

    # LTT block present and self-consistent
    ltt = m["ltt"]
    assert set(ltt) >= {"alpha", "lambda_hat", "coverage", "selective_risk", "risk_within_budget"}
    assert 0.0 <= ltt["coverage"] <= 1.0

    assert np.isfinite(m["accuracy_id"]) and np.isfinite(m["ece_geometry"])
    json.dumps(m)  # must be ledger-serialisable


def test_folds_are_independently_sampled():
    # The three ID folds draw from distinct seeding substreams, so they are not the same batch.
    cfg = load_config("configs/smoke.toml")
    task = build_task(cfg.run.task)
    batches = [task.sample(numpy_rng(cfg.run.seed, 20, s), cfg.eval.n_eval, "id").x
               for s in (0, 2)]
    assert not np.array_equal(batches[0], batches[1])
