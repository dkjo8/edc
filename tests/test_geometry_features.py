"""Phase-2 geometry features: exact values on hand-built trajectories, shape/finiteness on a
real (untrained) tiny model, determinism, and batched-curvature correctness.

Offline + CPU-only (conftest forces JAX_PLATFORMS=cpu). No training needed: the geometry
features are a pure function of a TrajectoryRecord, so we exercise them on a freshly-built
model plus small hand-constructed records.
"""

import numpy as np

from edc.config import load_config
from edc.energy import mlp_ebm
from edc.eval.metrics import single_feature_auroc
from edc.geometry.basin import basin_features
from edc.geometry.curvature import (
    batched_curvature,
    energy_at,
    hutchinson_trace,
    lambda_max,
)
from edc.geometry.dynamics import dynamics_features
from edc.geometry.energy_stats import energy_features
from edc.geometry.features import geometry_features
from edc.inference import restarts
from edc.inference.trajectory import TrajectoryRecord
from edc.seeding import numpy_rng, root_key
from edc.tasks.arithmetic import ArithmeticTask

N_EXPECTED_FEATURES = 14  # basin 3 + energy 3 + curvature 4 + dynamics 4


def _toy_traj(pred, z_star, energies, grad_norms, n_classes):
    """Build a TrajectoryRecord from NumPy arrays (logits/h_x are placeholders for the
    basin/energy/dynamics numpy features, which never touch them)."""
    pred = np.asarray(pred)
    B, K = pred.shape
    logits = np.zeros((B, K, n_classes))
    return TrajectoryRecord(
        z_star=np.asarray(z_star, dtype=float),
        energies=np.asarray(energies, dtype=float),
        grad_norms=np.asarray(grad_norms, dtype=float),
        logits=logits,
        pred=pred,
        h_x=np.zeros((B, 4)),
    )


# ------------------------------- basin / energy / dynamics exact values -----------------------


def test_basin_unanimous_vs_split():
    # Two inputs: input 0 all restarts agree (class 2); input 1 splits 2-2.
    pred = np.array([[2, 2, 2, 2], [0, 0, 1, 1]])
    z = np.zeros((2, 4, 3))            # identical endpoints => zero dispersion
    feats, names = basin_features(_toy_traj(pred, z, np.zeros((2, 4, 2)), np.zeros((2, 4, 1)), 3))
    assert names == ["basin/rho", "basin/entropy", "basin/dispersion"]

    rho, entropy, dispersion = feats[:, 0], feats[:, 1], feats[:, 2]
    assert rho[0] == 1.0 and entropy[0] == 0.0            # unanimous
    assert np.isclose(rho[1], 0.5)
    assert np.isclose(entropy[1], np.log(2))             # 2-2 split -> ln 2 nats
    assert np.allclose(dispersion, 0.0)


def test_basin_dispersion_grows_with_spread():
    pred = np.zeros((1, 2), dtype=int)
    z_close = np.array([[[0.0, 0.0], [0.1, 0.0]]])
    z_far = np.array([[[0.0, 0.0], [3.0, 4.0]]])          # pairwise distance 5
    zeros = (np.zeros((1, 2, 2)), np.zeros((1, 2, 1)))
    d_close = basin_features(_toy_traj(pred, z_close, *zeros, 1))[0][0, 2]
    d_far = basin_features(_toy_traj(pred, z_far, *zeros, 1))[0][0, 2]
    assert np.isclose(d_close, 0.1)
    assert np.isclose(d_far, 5.0)
    assert d_far > d_close


def test_energy_stats_order():
    # terminal energy = energies[..., -1]. Input 0 restarts end at {1, 3}; input 1 at {2, 2}.
    energies = np.array([[[9, 1], [9, 3]], [[9, 2], [9, 2]]], dtype=float)  # (B=2, K=2, T+1=2)
    feats, names = energy_features(_toy_traj(np.zeros((2, 2), int), np.zeros((2, 2, 1)),
                                             energies, np.zeros((2, 2, 1)), 1))
    assert names == ["energy/mean", "energy/min", "energy/std"]
    emean, emin, estd = feats[:, 0], feats[:, 1], feats[:, 2]
    assert np.allclose(emean, [2.0, 2.0]) and np.allclose(emin, [1.0, 2.0])
    assert np.all(emin <= emean + 1e-9)
    assert np.isclose(estd[1], 0.0)                       # identical -> zero spread


def test_dynamics_values():
    # One input, one restart. Energy strictly decreasing 4->3->2->1 (T=3 steps): monotonic=1.
    energies = np.array([[[4.0, 3.0, 2.0, 1.0]]])         # (1, 1, 4)
    grad_norms = np.array([[[0.5, 0.4, 1e-4]]])           # converges at step 2 (< tol)
    traj = _toy_traj(np.zeros((1, 1), int), np.zeros((1, 1, 1)), energies, grad_norms, 1)
    feats, names = dynamics_features(traj, grad_tol=1e-3)
    assert names == ["dynamics/steps", "dynamics/monotonic", "dynamics/drop", "dynamics/residual"]
    steps, monotonic, drop, residual = feats[0]
    assert np.isclose(steps, 2 / 3)                       # first below-tol step is index 2, /T
    assert np.isclose(monotonic, 1.0)
    assert np.isclose(drop, 3.0)                          # 4 - 1
    assert np.isclose(residual, 1e-4)                     # terminal grad norm


def test_dynamics_steps_defaults_to_T_when_never_converged():
    grad_norms = np.array([[[1.0, 1.0, 1.0]]])            # never below tol
    traj = _toy_traj(np.zeros((1, 1), int), np.zeros((1, 1, 1)),
                     np.zeros((1, 1, 4)), grad_norms, 1)
    steps = dynamics_features(traj, grad_tol=1e-3)[0][0, 0]
    assert np.isclose(steps, 1.0)                         # T/T


# ------------------------------- single-feature AUROC -----------------------------------------


def test_single_feature_auroc():
    correct = np.array([True, True, False, False])
    assert single_feature_auroc(np.array([3.0, 4.0, 1.0, 2.0]), correct) == 1.0   # ordered
    assert single_feature_auroc(np.array([1.0, 2.0, 3.0, 4.0]), correct) == 0.0   # reversed
    assert single_feature_auroc(np.array([1.0, 1.0, 1.0, 1.0]), correct) == 0.5   # all ties
    assert single_feature_auroc(np.array([1.0, 2.0, 3.0]), np.array([True, True, True])) == 0.5


def test_auroc_near_half_on_noise():
    rng = np.random.default_rng(0)
    feature = rng.standard_normal(2000)
    correct = rng.integers(0, 2, size=2000).astype(bool)
    assert abs(single_feature_auroc(feature, correct) - 0.5) < 0.05


# ------------------------------- batched curvature --------------------------------------------


def test_batched_curvature_matches_quadratic():
    d, n = 5, 3
    A = np.random.default_rng(0).standard_normal((d, d)).astype(np.float32)
    A = A @ A.T + np.eye(d, dtype=np.float32)             # SPD

    def energy(_params, _h, z):                           # ignores params/context
        return 0.5 * np.einsum("ni,ij,nj->n", np.asarray(z), A, np.asarray(z))

    import jax.numpy as jnp

    def jax_energy(_params, _h, z):
        return 0.5 * jnp.einsum("ni,ij,nj->n", z, jnp.asarray(A), z)

    zs = jnp.asarray(np.random.default_rng(1).standard_normal((n, d)).astype(np.float32))
    contexts = jnp.zeros((n, 1))
    lmax, trace = batched_curvature(jax_energy, {}, contexts, zs, root_key(3),
                                    iters=60, n_probes=256)
    top = float(np.linalg.eigvalsh(A)[-1])
    assert np.allclose(np.asarray(lmax), top, rtol=2e-2)  # Hessian is constant A for every z
    assert np.allclose(np.asarray(trace), float(np.trace(A)), rtol=0.15)
    _ = energy  # numpy reference kept for documentation


def test_batched_curvature_matches_per_particle_loop():
    # Batched wrapper must equal the per-particle primitives with the same derived keys.
    import jax

    cfg = load_config("configs/smoke.toml")
    task = ArithmeticTask()
    params, fns, _ = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, root_key(0))

    import jax.numpy as jnp

    batch = task.sample(numpy_rng(0, 1), 3, "id")
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, root_key(0))

    B, K, dlat = np.asarray(traj.z_star).shape
    contexts = jnp.asarray(np.repeat(np.asarray(traj.h_x), K, axis=0))
    zs = jnp.asarray(np.asarray(traj.z_star).reshape(B * K, dlat))
    key = root_key(7)

    lmax_b, trace_b = batched_curvature(fns.energy, params, contexts, zs, key)

    N = contexts.shape[0]
    keys = jax.random.split(key, N)
    lmax_ref, trace_ref = [], []
    for i in range(N):
        k_lam, k_tr = jax.random.split(keys[i])
        f = energy_at(fns.energy, params, contexts[i])
        lmax_ref.append(float(lambda_max(f, zs[i], k_lam)))
        trace_ref.append(float(hutchinson_trace(f, zs[i], k_tr)))
    assert np.allclose(np.asarray(lmax_b), lmax_ref, atol=1e-4)
    assert np.allclose(np.asarray(trace_b), trace_ref, atol=1e-4)


# ------------------------------- full assembly on a real model --------------------------------


def _real_traj(seed=0, n=8):
    import jax.numpy as jnp

    cfg = load_config("configs/smoke.toml")
    task = ArithmeticTask()
    params, fns, _ = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, root_key(seed))
    batch = task.sample(numpy_rng(seed, 1), n, "id")
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, root_key(seed))
    return traj, fns, params, cfg


def test_geometry_features_shape_and_names():
    traj, fns, params, cfg = _real_traj(n=8)
    feats, names = geometry_features(traj, fns, params, root_key(1),
                                     grad_tol=cfg.inference.grad_tol)
    assert feats.shape == (8, N_EXPECTED_FEATURES)
    assert len(names) == N_EXPECTED_FEATURES
    assert len(set(names)) == N_EXPECTED_FEATURES        # unique
    assert np.all(np.isfinite(feats))


def test_geometry_features_deterministic():
    traj, fns, params, cfg = _real_traj(n=6)
    f1, _ = geometry_features(traj, fns, params, root_key(2), grad_tol=cfg.inference.grad_tol)
    f2, _ = geometry_features(traj, fns, params, root_key(2), grad_tol=cfg.inference.grad_tol)
    assert np.array_equal(f1, f2)


# ------------------------------- Phase 4k: opt-in richer geometry ------------------------------

N_RICHER_FEATURES = N_EXPECTED_FEATURES + 8 + 3  # + spectrum 8 + connect 3 = 25


def test_richer_appends_spectrum_and_connect_without_touching_base():
    traj, fns, params, cfg = _real_traj(n=8)
    base, base_names = geometry_features(traj, fns, params, root_key(1),
                                         grad_tol=cfg.inference.grad_tol)
    rich, rich_names = geometry_features(traj, fns, params, root_key(1),
                                         grad_tol=cfg.inference.grad_tol, richer=True)
    # richer==False stays exactly 14; richer==True appends the two new groups after the base 14.
    assert base.shape == (8, N_EXPECTED_FEATURES)
    assert rich.shape == (8, N_RICHER_FEATURES)
    assert rich_names[:N_EXPECTED_FEATURES] == base_names        # base names/order unchanged
    assert np.array_equal(rich[:, :N_EXPECTED_FEATURES], base)   # base values byte-identical
    assert len(set(rich_names)) == N_RICHER_FEATURES            # unique
    assert np.all(np.isfinite(rich))
    prefixes = {n.split("/")[0] for n in rich_names[N_EXPECTED_FEATURES:]}
    assert prefixes == {"spectrum", "connect"}
