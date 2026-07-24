"""Training objective for the energy reasoner (Phase 1: supervised basin center).

The energy is a learned bowl centred at ``c(h_x)`` plus a bounded correction
(``edc.energy.mlp_ebm``), so Langevin descent converges toward ``c(h_x)``. Training therefore
only needs to make the decoder read the correct answer off that basin:

* **decode CE** — ``decode(c(h_x))`` and ``decode(c + small noise)`` both classify to ``y``, so
  the whole basin (not just its exact centre, where descent lands up to Langevin noise) decodes
  correctly.
* **spread reg** — a mild penalty keeps ``||c||`` in the range the z-initialisation can reach,
  so descent from ``N(0, init_scale^2 I)`` actually arrives.

No codebook and no score matching: the bowl supplies the gradient field for free. Phase 4
swaps this for a fully-learned IRED-style landscape (annealed denoising score matching), where
the training objective becomes the interesting part.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def loss_fn(params, model, x, y, key, neg_noise: float, center_reg: float = 1e-3):
    from edc.energy.mlp_ebm import EnergyReasoner

    def encode(p, xx):
        return model.apply(p, xx, method=EnergyReasoner.encode)

    def center(p, h):
        return model.apply(p, h, method=EnergyReasoner.center)

    def decode(p, z):
        return model.apply(p, z, method=EnergyReasoner.decode)

    h_x = encode(params, x)
    c = center(params, h_x)                                       # (n, d) basin centre
    z_near = c + neg_noise * jax.random.normal(key, c.shape)      # a point the descent may land

    ce = 0.5 * (
        optax.softmax_cross_entropy_with_integer_labels(decode(params, c), y).mean()
        + optax.softmax_cross_entropy_with_integer_labels(decode(params, z_near), y).mean()
    )
    reg = center_reg * jnp.mean(jnp.sum(c**2, axis=-1))

    total = ce + reg
    metrics = {
        "loss": total,
        "ce": ce,
        "center_norm": jnp.sqrt(jnp.mean(jnp.sum(c**2, axis=-1))),
    }
    return total, metrics
