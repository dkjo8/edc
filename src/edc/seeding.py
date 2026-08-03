"""Deterministic randomness for EDC.

All randomness in the project flows through here so that a run is an exact function
of its integer ``seed``. This is what makes "regenerate figure N from seed + config"
a pure, cached pipeline (see ``.claude/CLAUDE.md`` invariant 4 and 6).

We keep two layers:

* **JAX PRNG keys** — for the model, Langevin restarts, and any XLA-side sampling.
  Per-restart keys are derived by ``fold_in`` so restart ``k``'s noise is a pure
  function of ``(seed, k)``.
* **NumPy Generators** — for host-side data generation and conformal split shuffling,
  where JAX arrays would be awkward.

Nothing outside this module should call ``jax.random.PRNGKey`` or ``np.random.*`` directly.
"""

from __future__ import annotations

import jax
import numpy as np


def root_key(seed: int) -> jax.Array:
    """Root JAX PRNG key for a run."""
    return jax.random.PRNGKey(int(seed))


def restart_keys(seed: int, k_restarts: int) -> jax.Array:
    """A ``(k_restarts,)`` batch of independent keys, one per Langevin restart.

    Uses ``fold_in`` rather than ``split`` so that restart ``k`` has the same key
    regardless of ``k_restarts`` — adding restarts never perturbs existing ones,
    which keeps ablations over ``K`` clean and reproducible.
    """
    base = root_key(seed)
    return jax.vmap(lambda k: jax.random.fold_in(base, k))(np.arange(int(k_restarts)))


def numpy_rng(seed: int, *stream: int) -> np.random.Generator:
    """A NumPy ``Generator`` for host-side work (data, conformal splits).

    ``stream`` lets callers derive independent, named substreams from one seed,
    e.g. ``numpy_rng(seed, 1)`` for data and ``numpy_rng(seed, 2)`` for the calib split,
    without them ever sharing state.
    """
    return np.random.default_rng([int(seed), *[int(s) for s in stream]])


def fold(key: jax.Array, i: int) -> jax.Array:
    """Deterministically derive a child key indexed by ``i``."""
    return jax.random.fold_in(key, int(i))


def member_seeds(seed: int, m: int) -> np.ndarray:
    """``m`` distinct integer seeds for deep-ensemble members, derived from ``seed``. [Phase 4m]

    Drawn from a dedicated substream (``stream 7``) so the whole ensemble stays a pure function of
    the run ``seed`` (invariant 4) and two different base seeds get disjoint member sets (no leakage
    of one seed's members into another's ensemble). Each returned seed is fed to ``train`` as a
    fresh ``cfg.run.seed`` to give an independently-initialised, independently-sampled model.
    """
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    return numpy_rng(seed, 7).integers(0, 2**31 - 1, size=int(m))
