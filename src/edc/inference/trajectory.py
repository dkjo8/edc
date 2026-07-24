"""Records produced by K-restart inference.

Shapes use ``B`` = number of inputs, ``K`` = restarts, ``T`` = descent steps, ``d`` = latent
dim, ``C`` = classes. These arrays are the *raw material* the Phase-2 geometry features are
computed from — we deliberately keep the full descent, not just the endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Array = Any


@dataclass
class TrajectoryRecord:
    z_star: Array         # (B, K, d)   final latent per restart
    energies: Array       # (B, K, T+1) energy at each step incl. initial
    grad_norms: Array     # (B, K, T)   ||grad_z E|| at each step
    logits: Array         # (B, K, C)   decoder logits at z_star
    pred: Array           # (B, K)      argmax class per restart

    @property
    def terminal_energy(self) -> Array:
        return self.energies[..., -1]        # (B, K)

    @property
    def terminal_grad_norm(self) -> Array:
        return self.grad_norms[..., -1]      # (B, K)
