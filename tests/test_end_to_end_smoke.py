"""Full Phase-1 path on CPU: train -> K-restart infer -> decode. Must stay well under 60s."""

import jax.numpy as jnp
import numpy as np

from edc.config import load_config
from edc.inference import restarts
from edc.seeding import numpy_rng, root_key
from edc.tasks.arithmetic import ArithmeticTask
from edc.train.train_ebm import train


def test_train_then_infer_produces_valid_accuracy():
    cfg = load_config("configs/smoke.toml")
    task = ArithmeticTask()

    params, fns, history = train(cfg, task)
    assert history["epochs"], "training produced no epoch metrics"
    # Loss should be finite and the contrastive term defined.
    last = history["epochs"][-1]
    assert np.isfinite(last["loss"])

    key = root_key(cfg.run.seed)
    batch = task.sample(numpy_rng(cfg.run.seed, 99), 128, "id")
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, key)
    pred, _ = restarts.best_of_n_energy(traj)
    acc = float(np.mean(task.evaluate(np.asarray(pred), batch.y)))
    assert 0.0 <= acc <= 1.0
