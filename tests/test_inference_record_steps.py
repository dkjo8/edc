"""solve(record_steps=True) populates a correct per-step decode trajectory; default leaves it None
and does not perturb the descent. Marked slow (builds a tiny model). Offline + CPU-only.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from edc.config import load_config
from edc.energy import mlp_ebm
from edc.inference import restarts
from edc.seeding import numpy_rng, root_key
from edc.tasks.arithmetic import ArithmeticTask

pytestmark = pytest.mark.slow


def _model():
    cfg = load_config("configs/smoke.toml")
    task = ArithmeticTask()
    params, fns, _ = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, root_key(0))
    x = jnp.asarray(task.sample(numpy_rng(0, 1), 6, "id").x)
    return cfg, fns, params, x


def test_record_steps_shape_and_final_consistency():
    cfg, fns, params, x = _model()
    traj = restarts.solve(fns, params, x, cfg, root_key(0), record_steps=True)
    B, K = np.asarray(traj.pred).shape
    T1 = np.asarray(traj.energies).shape[-1]
    sp = np.asarray(traj.step_pred)
    assert sp.shape == (B, K, T1)
    # the last recorded step is exactly the final decoded prediction
    assert np.array_equal(sp[:, :, -1], np.asarray(traj.pred))


def test_record_steps_off_is_none_and_nonintrusive():
    cfg, fns, params, x = _model()
    on = restarts.solve(fns, params, x, cfg, root_key(0), record_steps=True)
    off = restarts.solve(fns, params, x, cfg, root_key(0))
    assert off.step_pred is None
    # recording must not change the descent itself
    assert np.allclose(np.asarray(on.energies), np.asarray(off.energies))
    assert np.array_equal(np.asarray(on.pred), np.asarray(off.pred))
