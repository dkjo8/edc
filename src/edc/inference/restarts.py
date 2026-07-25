"""K stochastic restarts of the Langevin descent, batched over inputs.

Flattens ``(B inputs) x (K restarts)`` into ``N = B*K`` particles, runs one
``langevin_descent``, then reshapes back to ``(B, K, ...)`` and decodes each restart.
The result (a ``TrajectoryRecord``) is everything downstream geometry/selection needs.

Restart initialisations and noise streams derive from ``edc.seeding`` so a run is an exact
function of its seed (invariant 4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from edc.energy.base import ReasonerFns
from edc.inference.optimizer import langevin_descent
from edc.inference.trajectory import TrajectoryRecord


def solve(fns: ReasonerFns, params, x, cfg, key, k_restarts: int | None = None) -> TrajectoryRecord:
    """Run K-restart inference on a batch of raw inputs ``x`` (``(B, feature_dim)``)."""
    K = int(k_restarts if k_restarts is not None else cfg.inference.k_restarts)
    d = fns.latent_dim

    h_x = fns.encode(params, x)                          # (B, context_dim)
    B = h_x.shape[0]

    # Tile context to one row per particle: (B, K, dc) -> (N, dc).
    h_rep = jnp.reshape(jnp.repeat(h_x[:, None, :], K, axis=1), (B * K, -1))

    key_init, key_lan = jax.random.split(key)
    z0 = cfg.inference.init_scale * jax.random.normal(key_init, (B * K, d))

    z_final, energies, gnorms = langevin_descent(
        fns.energy, params, h_rep, z0, key_lan,
        step_size=cfg.inference.step_size,
        temperature=cfg.inference.temperature,
        steps=cfg.inference.steps,
    )

    logits = fns.decode(params, z_final)                 # (N, C)
    pred = jnp.argmax(logits, axis=-1)                   # (N,)

    def to_bk(a):
        return jnp.reshape(a, (B, K) + a.shape[1:])

    return TrajectoryRecord(
        z_star=to_bk(z_final),          # (B, K, d)
        energies=to_bk(energies.T),     # (N, T+1) -> (B, K, T+1)
        grad_norms=to_bk(gnorms.T),     # (N, T)   -> (B, K, T)
        logits=to_bk(logits),           # (B, K, C)
        pred=to_bk(pred),               # (B, K)
        h_x=h_x,                        # (B, dc)  one context per input (shared across restarts)
    )


def best_of_n_energy(traj: TrajectoryRecord):
    """EBT-style baseline answer: the restart with the lowest terminal energy.

    Returns ``(pred, chosen_restart)`` each of shape ``(B,)``. This is the *scalar-energy*
    selection that the geometry features must beat (the falsification baseline).
    """
    k_best = jnp.argmin(traj.terminal_energy, axis=1)    # (B,)
    b_idx = jnp.arange(traj.pred.shape[0])
    return traj.pred[b_idx, k_best], k_best


def majority_vote(traj: TrajectoryRecord):
    """Plurality decoded answer across restarts, shape ``(B,)``."""
    n_classes = traj.logits.shape[-1]
    counts = jax.nn.one_hot(traj.pred, n_classes).sum(axis=1)   # (B, C)
    return jnp.argmax(counts, axis=-1)
