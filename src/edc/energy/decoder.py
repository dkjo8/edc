"""Decoder g_psi: latent z -> class logits. (JAX/Flax)

A deliberately shallow read-out: the reasoning happens in the energy descent that lands
``z`` in a basin, not in the decoder. Keeping the decoder simple means the geometry of the
landscape (Phase 2) is what determines correctness, which is the whole thesis.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class Decoder(nn.Module):
    n_classes: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden_dim)(z))
        return nn.Dense(self.n_classes)(h)
