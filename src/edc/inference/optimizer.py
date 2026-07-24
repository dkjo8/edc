"""Langevin descent inner loop (JAX core).

Runs the update ``z_{t+1} = z_t - eta * grad_z E + sqrt(2*eta*tau) * noise`` for ``T`` steps
over a flat batch of ``N`` particles in parallel, via ``lax.scan`` (XLA-fused). Each particle
carries its own context ``h_x`` and its own PRNG stream, so this one function serves both the
per-input batch and the K restarts once they are flattened to ``N = B*K`` (see ``restarts``).

``tau > 0`` is required: with ``tau = 0`` every restart from the same basin collapses to the
same point and basin-agreement — the core geometry signal — degenerates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _per_particle_grad(energy, params, h_x):
    """grad_z of E, per particle. Sum over particles has zero cross terms, so the
    gradient of the sum equals the stacked per-particle gradients."""

    def total_energy(z):
        return jnp.sum(energy(params, h_x, z))

    return jax.grad(total_energy)


def langevin_descent(energy, params, h_x, z0, key, step_size, temperature, steps):
    """Descend ``N`` particles for ``steps`` steps.

    Args:
      energy: pure fn ``(params, h_x, z) -> (N,)`` energies.
      h_x: ``(N, context_dim)``.
      z0:  ``(N, d)`` initial latents.
      key: JAX PRNG key for the Langevin noise.

    Returns ``(z_final, energies, grad_norms)`` with
      ``energies``   shape ``(steps+1, N)`` (row 0 = initial),
      ``grad_norms`` shape ``(steps, N)``   (norm of the gradient used at each step).
    """
    grad_fn = _per_particle_grad(energy, params, h_x)
    noise_scale = jnp.sqrt(2.0 * step_size * temperature)

    def step(carry, _):
        z, k = carry
        k, sub = jax.random.split(k)
        g = grad_fn(z)                                   # (N, d)
        z_new = z - step_size * g + noise_scale * jax.random.normal(sub, z.shape)
        e_new = energy(params, h_x, z_new)               # (N,)
        gnorm = jnp.linalg.norm(g, axis=-1)              # (N,)
        return (z_new, k), (e_new, gnorm)

    e0 = energy(params, h_x, z0)                          # (N,)
    (z_final, _), (e_steps, gnorms) = jax.lax.scan(
        step, (z0, key), None, length=steps
    )
    energies = jnp.concatenate([e0[None, :], e_steps], axis=0)  # (steps+1, N)
    return z_final, energies, gnorms
