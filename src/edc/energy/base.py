"""Framework-agnostic interface to the energy-based reasoner.

The rest of the codebase (inference, geometry, training loops that don't touch params)
depends only on three pure functions, so the JAX core is swappable:

* ``encode(params, x)      -> h_x``
* ``energy(params, h_x, z) -> scalar``   (differentiable in ``z``; params fixed)
* ``decode(params, z)      -> logits``

``ReasonerFns`` bundles them. The concrete JAX/Flax implementation lives in
``edc.energy.mlp_ebm`` and builds one of these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Array = Any  # jax.Array at runtime; typed loosely to avoid importing jax here


@dataclass(frozen=True)
class ReasonerFns:
    """Pure apply-functions for a trained (or freshly-initialised) reasoner."""

    encode: Callable[[Any, Array], Array]          # (params, x)      -> h_x
    energy: Callable[[Any, Array, Array], Array]   # (params, h_x, z) -> scalar
    decode: Callable[[Any, Array], Array]          # (params, z)      -> logits
    latent_dim: int
    n_classes: int
