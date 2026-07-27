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


def ired_loss_fn(params, model, x, y, key, noise_min: float, noise_max: float,
                 decode_weight: float, init_scale: float = 1.0, gap_weight: float = 1.0):
    """IRED-style denoising score matching toward a learned per-class latent codebook. [Phase 4g]

    Carves a genuinely learned, input-conditioned multi-basin landscape (no fixed bowl): the score
    ``-grad_z E`` is trained to denoise latents corrupted around the true class anchor ``mu_y`` back
    toward it, across a range of noise scales (annealing). Decode terms make the anchors — and the
    basins around them — read out the correct class, so K-restart descent lands answers *and* leaves
    cross-restart geometry that (hypothesis) carries signal beyond the decoder softmax.
    """
    from edc.energy.mlp_ebm import EnergyReasoner

    h_x = model.apply(params, x, method=EnergyReasoner.encode)
    anchors = model.apply(params, method=EnergyReasoner.anchors_all)       # (C, d)
    mu_y = anchors[y]                                                      # (n, d)
    n, d = mu_y.shape

    k_sig, k_eps, k_rand = jax.random.split(key, 3)
    # log-uniform noise scale per example (annealed DSM), then Gaussian corruption.
    log_sigma = jax.random.uniform(
        k_sig, (n, 1), minval=jnp.log(noise_min), maxval=jnp.log(noise_max))
    sigma = jnp.exp(log_sigma)
    eps = jax.random.normal(k_eps, mu_y.shape)
    z_tilde = mu_y + sigma * eps

    def energy_at(z):
        return model.apply(params, h_x, z, method=EnergyReasoner.energy)

    grad_z = jax.grad(lambda z: jnp.sum(energy_at(z)))(z_tilde)            # (n, d), per-example
    # DSM: score -grad E should match the Gaussian score -eps/sigma  =>  sigma*grad E ~ eps.
    dsm = jnp.mean(jnp.sum((sigma * grad_z - eps) ** 2, axis=-1))

    # Reachability: the true-class anchor must be lower-energy than a random z0-like point, so
    # descent from N(0, init_scale^2) flows toward mu_y (DSM alone only shapes local basins).
    z_rand = init_scale * jax.random.normal(k_rand, mu_y.shape)
    gap = jax.nn.softplus(energy_at(mu_y) - energy_at(z_rand) + 1.0).mean()

    # Decode: each anchor reads out its own class; the basin around mu_y reads out y.
    ce_anchor = optax.softmax_cross_entropy_with_integer_labels(
        model.apply(params, anchors, method=EnergyReasoner.decode), jnp.arange(anchors.shape[0])
    ).mean()
    ce_near = optax.softmax_cross_entropy_with_integer_labels(
        model.apply(params, z_tilde, method=EnergyReasoner.decode), y
    ).mean()

    total = dsm + decode_weight * (ce_anchor + ce_near) + gap_weight * gap
    metrics = {
        "loss": total,
        "dsm": dsm,
        "gap": gap,
        "ce_anchor": ce_anchor,
        "ce_near": ce_near,
        "anchor_norm": jnp.sqrt(jnp.mean(jnp.sum(anchors**2, axis=-1))),
    }
    return total, metrics
