"""Modular arithmetic reasoning (ported from EBRM ``arithmetic_reasoning.jl``).

Problem: sum of ``k`` operands, each in ``[0, max_operand]``, taken mod ``modulus``.
The answer label is ``(sum) % modulus`` in ``[0, modulus)``.

Features are a fixed-size one-hot block: ``max_operands`` slots of ``(max_operand+1)``
one-hot each; unused slots (for smaller ``k``) are zero. This keeps ``feature_dim`` constant
so the in-distribution split (``n_operands``) and the OOD split (``ood_n_operands`` > it) share
one model. More operands => more carries to track => harder => lower accuracy, which is
exactly the partially-competent regime where selective prediction has error to remove.
"""

from __future__ import annotations

import numpy as np

from edc.registry import register_task
from edc.tasks.base import Batch

_MAX_OPERANDS = 8  # padding capacity; both ID and OOD operand counts must be <= this


class ArithmeticTask:
    name = "arithmetic"

    def __init__(
        self,
        modulus: int = 10,
        n_operands: int = 3,
        max_operand: int = 9,
        ood_n_operands: int = 5,
        ood_max_operand: int | None = None,
        modular: bool = False,
    ) -> None:
        if max(n_operands, ood_n_operands) > _MAX_OPERANDS:
            raise ValueError(f"operand count exceeds capacity {_MAX_OPERANDS}")
        self.modulus = modulus
        self.n_operands = n_operands
        self.max_operand = max_operand
        self.ood_n_operands = ood_n_operands
        # OOD shift 1: more operands (ood_n_operands). OOD shift 2: larger operand magnitudes
        # (ood_max_operand > max_operand) — a covariate shift that keeps the reasoning identical
        # and the low-sum region in-distribution, so accuracy degrades *gracefully* rather than
        # collapsing. Defaults to max_operand (no magnitude shift).
        self.ood_max_operand = max_operand if ood_max_operand is None else ood_max_operand
        # modular=True is the harder "grokking" variant (answer = sum mod modulus);
        # modular=False (default) is plain sum classification — smooth in the inputs and the
        # right demonstrator that descent lands in the correct basin. Both share feature_dim.
        self.modular = modular
        # Encoding width and class count must cover both splits so one model handles ID + OOD.
        self._slot_max = max(max_operand, self.ood_max_operand)
        self.n_classes = modulus if modular else _MAX_OPERANDS * self._slot_max + 1
        self.feature_dim = _MAX_OPERANDS * (self._slot_max + 1)

    def _k(self, split: str) -> int:
        return self.ood_n_operands if split == "ood" else self.n_operands

    def _max(self, split: str) -> int:
        return self.ood_max_operand if split == "ood" else self.max_operand

    def sample(self, rng: np.random.Generator, n: int, split: str = "id") -> Batch:
        k = self._k(split)
        operands = rng.integers(0, self._max(split) + 1, size=(n, k))
        total = operands.sum(axis=1)
        y = (total % self.modulus) if self.modular else total

        slot = self._slot_max + 1
        x = np.zeros((n, self.feature_dim), dtype=np.float32)
        for j in range(k):
            cols = j * slot + operands[:, j]
            x[np.arange(n), cols] = 1.0
        return Batch(x=x, y=y.astype(np.int64))

    def evaluate(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(pred) == np.asarray(y)

    def difficulty(self, batch: Batch) -> np.ndarray:
        # Number of active operand slots ~ proxy for carry complexity.
        slot = self._slot_max + 1
        blocks = batch.x.reshape(batch.x.shape[0], _MAX_OPERANDS, slot)
        return blocks.max(axis=2).sum(axis=1).astype(np.float32)


@register_task("arithmetic")
def _build(**kwargs) -> ArithmeticTask:
    # Accept keys from the [task.arithmetic] TOML block; ignore unrelated ones.
    known = {"modulus", "n_operands", "max_operand", "ood_n_operands", "ood_max_operand",
             "modular"}
    return ArithmeticTask(**{k: v for k, v in kwargs.items() if k in known})
