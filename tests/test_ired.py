"""IRED opt-in learned-landscape training (Phase 4g): the contrastive+stationarity objective trains
without error and is non-breaking (the default basin_center path and energy form are unchanged).
Offline + CPU-only. Reasoning accuracy (which needs many epochs + the annealed sampler) is exercised
by the experiment config, not asserted here — this guards the plumbing.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from edc.config import load_config, load_from_dict
from edc.energy import mlp_ebm
from edc.inference import restarts
from edc.registry import build_task
from edc.seeding import numpy_rng, root_key


def _ired_cfg():
    d = {k: (dict(v) if isinstance(v, dict) else v)
         for k, v in load_config("configs/smoke.toml").to_dict().items()}
    d["train"] = {**d["train"], "objective": "ired", "epochs": 2}
    return load_from_dict(d)


def test_default_objective_is_basin_center_and_bowl():
    cfg = load_config("configs/smoke.toml")
    assert cfg.train.objective == "basin_center"
    _, _, model = mlp_ebm.build(cfg, n_classes=5, feature_dim=8, key=root_key(0))
    assert model.energy_form == "bowl"                       # non-breaking default


def test_ired_build_uses_learned_energy_and_anchors():
    cfg = _ired_cfg()
    _, _, model = mlp_ebm.build(cfg, n_classes=5, feature_dim=8, key=root_key(0))
    assert model.energy_form == "learned"


@pytest.mark.slow
def test_ired_trains_and_infers_without_error():
    from edc.train.train_ebm import train

    cfg = _ired_cfg()
    task = build_task("arithmetic")
    params, fns, history = train(cfg, task)
    last = history["epochs"][-1]
    assert {"loss", "contrast", "stat", "e_pos", "e_neg", "ce_anchor"} <= set(last)
    assert all(np.isfinite(v) for v in last.values())        # trains without NaN/inf

    batch = task.sample(numpy_rng(0, 1), 8, "id")
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, root_key(0))
    pred, _ = restarts.best_of_n_energy(traj)
    pred = np.asarray(pred)
    assert pred.shape == (8,) and pred.min() >= 0 and pred.max() < task.n_classes
