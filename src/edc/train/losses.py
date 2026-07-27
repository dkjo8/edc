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
    """IREM-style contrastive learned-landscape training toward a per-class latent codebook. [4g]

    Carves an input-conditioned multi-basin energy (no fixed bowl) so that for input ``x`` the
    true-class anchor ``mu_y`` is a *reachable local minimum*: (i) **contrastive** — ``E(mu_y)`` is
    pushed below the wrong-class anchor and below a random ``z0``-like point by a margin, so descent
    from ``N(0, init_scale^2)`` flows toward it; (ii) **stationarity** — ``||grad_z E(mu_y)||``
    is driven to 0 so ``mu_y`` is an attractor descent settles into (a far easier double-grad target
    than full denoising score matching, which did not fit); (iii) **decode** — anchors and their
    basins read out the correct class. Cross-restart geometry then (hypothesis) carries signal
    beyond the decoder softmax. ``noise_min`` sets the basin-decode spread; ``noise_max`` is unused.
    """
    from edc.energy.mlp_ebm import EnergyReasoner

    _ = noise_max
    h_x = model.apply(params, x, method=EnergyReasoner.encode)
    anchors = model.apply(params, method=EnergyReasoner.anchors_all)       # (C, d)
    n_classes = anchors.shape[0]
    mu_y = anchors[y]                                                      # (n, d)

    k_neg, k_rand, k_near = jax.random.split(key, 3)

    def energy_at(z):
        return model.apply(params, h_x, z, method=EnergyReasoner.energy)   # (n,)

    e_pos = energy_at(mu_y)
    offset = jax.random.randint(k_neg, (mu_y.shape[0],), 1, n_classes)     # a different class
    mu_neg = anchors[(y + offset) % n_classes]
    e_neg = energy_at(mu_neg)
    z_rand = init_scale * jax.random.normal(k_rand, mu_y.shape)
    e_rand = energy_at(z_rand)

    margin = 1.0
    contrast = (jax.nn.softplus(e_pos - e_neg + margin).mean()
                + jax.nn.softplus(e_pos - e_rand + margin).mean())
    reg = 0.1 * jnp.mean(e_pos**2)                        # bound energies (anti-collapse)

    # Stationarity: grad_z E at the anchor -> 0, so mu_y is a local minimum descent settles into.
    grad_mu = jax.grad(lambda z: jnp.sum(energy_at(z)))(mu_y)              # (n, d)
    stat = jnp.mean(jnp.sum(grad_mu**2, axis=-1))

    ce_anchor = optax.softmax_cross_entropy_with_integer_labels(
        model.apply(params, anchors, method=EnergyReasoner.decode), jnp.arange(n_classes)).mean()
    z_near = mu_y + noise_min * jax.random.normal(k_near, mu_y.shape)
    ce_near = optax.softmax_cross_entropy_with_integer_labels(
        model.apply(params, z_near, method=EnergyReasoner.decode), y).mean()

    total = contrast + reg + gap_weight * stat + decode_weight * (ce_anchor + ce_near)
    metrics = {
        "loss": total,
        "contrast": contrast,
        "stat": stat,
        "e_pos": e_pos.mean(),
        "e_neg": e_neg.mean(),
        "ce_anchor": ce_anchor,
        "ce_near": ce_near,
    }
    return total, metrics
