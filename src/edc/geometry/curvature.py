"""Local curvature of the energy landscape at the inference optimum. (JAX core)

The technical crux of the method. We characterise the basin an EBM reasoner descends into by
the Hessian of the energy **with respect to the latent** ``H = d^2/dz^2 E_theta(h_x, z*)`` —
per input, at fixed weights. This is a *different object* from the loss-vs-weights curvature
that calibration/sharpness papers study (see ``docs/RELATED_WORK.md``), and it is the object
we claim is a calibratable correctness signal.

We never materialise ``H`` (it is ``d x d`` per particle). Everything is built on one
Hessian-vector product via forward-over-reverse autodiff::

    hvp(f, z, v) = d/deps grad f(z + eps v) |_{eps=0} = jvp(grad f, (z,), (v,))[1]

``lambda_max`` (sharpness) comes from power iteration on that operator; ``trace`` (basin
volume proxy) from a Hutchinson estimator. Both are the raw features Phase 2 assembles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def hvp(f, z, v):
    """Hessian-vector product of scalar ``f`` at ``z`` with vector ``v`` (same shape as ``z``)."""
    return jax.jvp(jax.grad(f), (z,), (v,))[1]


def lambda_max(f, z, key, iters: int = 12) -> jnp.ndarray:
    """Top Hessian eigenvalue at ``z`` via power iteration (sharpness of the basin)."""
    v = jax.random.normal(key, z.shape, dtype=z.dtype)
    v = v / (jnp.linalg.norm(v) + 1e-12)
    lam = jnp.array(0.0)
    for _ in range(iters):
        Hv = hvp(f, z, v)
        lam = jnp.vdot(v, Hv)
        nrm = jnp.linalg.norm(Hv)
        v = Hv / (nrm + 1e-12)
    return lam


def hutchinson_trace(f, z, key, n_probes: int = 8) -> jnp.ndarray:
    """Hutchinson estimate of ``tr(H)`` (a wide/flat basin => small trace)."""
    keys = jax.random.split(key, n_probes)

    def one(k):
        r = jax.random.rademacher(k, z.shape, dtype=z.dtype)
        return jnp.vdot(r, hvp(f, z, r))

    return jnp.mean(jax.vmap(one)(keys))


def energy_at(energy, params, h_x_row):
    """Curry the energy into a scalar function of a single latent ``z`` (for one particle)."""
    def f(z):
        # energy expects batched args; add/remove a leading axis to keep it scalar.
        return jnp.squeeze(energy(params, h_x_row[None, :], z[None, :]))

    return f


def batched_curvature(energy, params, contexts, zs, key, iters: int = 12, n_probes: int = 8):
    """Per-particle ``(lambda_max, tr(H))`` for a flat batch of ``N`` particles.

    This is the JAX-side entry point the (plain-NumPy) Phase-2 feature assembler calls, so all
    per-input Hessian-vector products and their ``vmap`` over restarts stay confined to the JAX
    core (invariant 1). ``contexts`` is ``(N, dc)`` and ``zs`` is ``(N, d)`` — flattened
    ``B*K`` particles exactly as ``inference.restarts.solve`` lays them out. Each particle gets
    its own PRNG stream (one key for power iteration, one for the Hutchinson probes) so the
    estimate is a pure function of ``key``. Returns ``(lmax (N,), trace (N,))``.
    """
    n = zs.shape[0]
    keys = jax.random.split(key, n)

    def one(h_row, z_row, k):
        k_lam, k_tr = jax.random.split(k)
        f = energy_at(energy, params, h_row)
        return (lambda_max(f, z_row, k_lam, iters=iters),
                hutchinson_trace(f, z_row, k_tr, n_probes=n_probes))

    return jax.vmap(one)(contexts, zs, keys)
