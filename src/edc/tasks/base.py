"""Task interface for EDC reasoning families.

A task turns a synthetic reasoning problem into fixed-shape float features (the reasoner's
input ``x``) and an integer answer label. Feature shape must be constant across the
in-distribution and OOD splits so the same model handles both — that is how we get the
OOD generalization test that selective prediction needs (see ``docs/RESEARCH_PLAN.md``).

Everything here is pure NumPy: no JAX in the task layer (invariant 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Batch:
    """A batch of problems.

    ``x``: ``(n, feature_dim)`` float32 features.
    ``y``: ``(n,)`` int answer labels in ``[0, n_classes)``.
    """

    x: np.ndarray
    y: np.ndarray


@runtime_checkable
class Task(Protocol):
    name: str
    feature_dim: int
    n_classes: int

    def sample(self, rng: np.random.Generator, n: int, split: str = "id") -> Batch:
        """Draw ``n`` problems. ``split`` in {"id", "ood"}."""
        ...

    def evaluate(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Elementwise correctness (bool array)."""

    def difficulty(self, batch: Batch) -> np.ndarray:
        """A scalar per problem (higher = harder); used for diagnostics/stratification."""
