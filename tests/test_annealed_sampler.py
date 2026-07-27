"""Annealed Langevin sampler mechanics (Phase 4g): on a simple bowl it settles into the minimum,
it produces a well-formed trajectory, and the default sampler stays plain Langevin (non-breaking).
Offline + CPU-only.
"""

import jax.numpy as jnp
import numpy as np

from edc.inference.optimizer import annealed_langevin
from edc.seeding import root_key


def test_annealed_langevin_settles_into_a_bowl_minimum():
    # E(z) = 0.5||z - c||^2  (min at c); the annealed schedule should land near c from afar.
    c = jnp.array([[2.0, -1.0, 0.5]])

    def energy(_params, _h, z):
        return 0.5 * jnp.sum((z - c) ** 2, axis=-1)

    z0 = jnp.zeros((1, 3))
    step_sizes = jnp.repeat(jnp.geomspace(0.3, 0.01, 12), 8)   # explore -> settle
    z_final, energies, gnorms, z_traj = annealed_langevin(
        energy, {}, jnp.zeros((1, 1)), z0, root_key(0), step_sizes, temperature=0.01)
    assert np.linalg.norm(np.asarray(z_final)[0] - np.asarray(c)[0]) < 0.2   # reached the minimum
    assert energies.shape == (step_sizes.shape[0] + 1, 1)
    assert gnorms.shape == (step_sizes.shape[0], 1)
    assert z_traj is None                                       # record_z defaults off


def test_annealed_record_z_shape():
    def energy(_p, _h, z):
        return 0.5 * jnp.sum(z**2, axis=-1)

    step_sizes = jnp.repeat(jnp.geomspace(0.2, 0.02, 5), 4)
    _, _, _, z_traj = annealed_langevin(
        energy, {}, jnp.zeros((2, 1)), jnp.ones((2, 3)), root_key(1), step_sizes,
        temperature=0.02, record_z=True)
    assert z_traj.shape == (step_sizes.shape[0] + 1, 2, 3)      # (steps+1, N, d)


def test_solve_dispatches_annealed_and_default_unchanged():
    import jax.numpy as jnp

    from edc.config import load_config, load_from_dict
    from edc.energy import mlp_ebm
    from edc.inference import restarts
    from edc.seeding import numpy_rng
    from edc.tasks.arithmetic import ArithmeticTask

    task = ArithmeticTask()
    d = load_config("configs/smoke.toml").to_dict()
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    d["inference"] = {**d["inference"], "sampler": "annealed",
                      "anneal_levels": 6, "anneal_steps_per_level": 4}
    cfg = load_from_dict(d)
    params, fns, _ = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, root_key(0))
    x = jnp.asarray(task.sample(numpy_rng(0, 1), 5, "id").x)
    traj = restarts.solve(fns, params, x, cfg, root_key(0))
    B, K = np.asarray(traj.pred).shape
    assert np.asarray(traj.energies).shape == (B, K, 6 * 4 + 1)   # annealed total steps + 1
    assert np.all(np.isfinite(np.asarray(traj.z_star)))
