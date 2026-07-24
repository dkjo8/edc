"""Encoder f_phi: problem features x -> context vector h_x. (JAX/Flax)"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class Encoder(nn.Module):
    context_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden_dim)(x))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.context_dim)(h)
