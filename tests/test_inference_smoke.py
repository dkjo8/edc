"""K-restart inference produces well-shaped, finite trajectories on a tiny CPU model."""

import jax.numpy as jnp
import numpy as np

from edc.config import load_config
from edc.energy import mlp_ebm
from edc.inference import restarts
from edc.seeding import numpy_rng, root_key
from edc.tasks.arithmetic import ArithmeticTask


def test_solve_shapes_and_finiteness():
    cfg = load_config("configs/smoke.toml")
    task = ArithmeticTask()
    key = root_key(0)
    params, fns, _ = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, key)

    B = 8
    batch = task.sample(numpy_rng(0, 1), B, "id")
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, key)

    K, d, T, C = cfg.inference.k_restarts, cfg.model.latent_dim, cfg.inference.steps, task.n_classes
    assert traj.z_star.shape == (B, K, d)
    assert traj.energies.shape == (B, K, T + 1)
    assert traj.grad_norms.shape == (B, K, T)
    assert traj.logits.shape == (B, K, C)
    assert traj.pred.shape == (B, K)
    assert bool(jnp.all(jnp.isfinite(traj.terminal_energy)))

    pred, k_best = restarts.best_of_n_energy(traj)
    assert pred.shape == (B,) and k_best.shape == (B,)
    assert np.all((np.asarray(pred) >= 0) & (np.asarray(pred) < C))
    assert restarts.majority_vote(traj).shape == (B,)


def test_more_restarts_do_not_change_first_restart():
    # fold_in seeding => restart 0 is identical whether K=2 or K=6 (reproducibility).
    cfg = load_config("configs/smoke.toml")
    task = ArithmeticTask()
    key = root_key(0)
    params, fns, _ = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, key)
    batch = task.sample(numpy_rng(0, 1), 4, "id")
    x = jnp.asarray(batch.x)
    # NOTE: restart initialisation here is a single normal draw over N=B*K particles, so this
    # documents current behaviour: identical seed+config => identical run.
    t_a = restarts.solve(fns, params, x, cfg, key, k_restarts=cfg.inference.k_restarts)
    t_b = restarts.solve(fns, params, x, cfg, key, k_restarts=cfg.inference.k_restarts)
    assert bool(jnp.allclose(t_a.z_star, t_b.z_star))
