"""The MLP energy-based reasoner and its builder. (JAX/Flax core)

Composes Flax modules into one parameter tree and exposes the pure functions of
``edc.energy.base.ReasonerFns`` plus a ``center`` helper used by training:

* ``encode(params, x)      -> h_x``
* ``center(params, h_x)    -> c``        predicted basin centre (training target)
* ``energy(params, h_x, z) -> scalar``   E_theta(h_x, z), differentiable in z
* ``decode(params, z)      -> logits``

**Energy form.** ``E(h_x, z) = 0.5 * ||z - c(h_x)||^2 + s * tanh(r(h_x, z))``: a learned bowl
centred at ``c(h_x)`` plus a *bounded* correction ``r`` (scale ``s``). The bowl guarantees that
Langevin descent converges toward ``c`` (so the base reasoner actually works and trains fast);
the bounded correction lets harder / OOD inputs grow competing minima and cross-restart
disagreement — the regime where the Phase-2 landscape-geometry features carry signal. Phase 4
replaces the bowl with a fully-learned IRED-style landscape.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from edc.energy.base import ReasonerFns
from edc.energy.decoder import Decoder
from edc.energy.encoder import Encoder

CORRECTION_SCALE = 1.0  # magnitude of the bounded non-convex correction to the bowl


class Correction(nn.Module):
    """Bounded scalar correction r(h_x, z) added to the quadratic bowl."""

    hidden_dim: int

    @nn.compact
    def __call__(self, h_x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        hz = jnp.concatenate([h_x, z], axis=-1)
        h = nn.relu(nn.Dense(self.hidden_dim)(hz))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return jnp.squeeze(nn.Dense(1)(h), axis=-1)


class EnergyReasoner(nn.Module):
    latent_dim: int
    hidden_dim: int
    context_dim: int
    n_classes: int
    energy_form: str = "bowl"   # "bowl" (Phase-1) or "learned" (IRED multi-basin, Phase 4g)
    ridge: float = 0.05         # ||z||^2 coefficient bounding the learned energy

    def setup(self) -> None:
        self.encoder = Encoder(self.context_dim, self.hidden_dim)
        self.center_head = nn.Dense(self.latent_dim)
        self.correction = Correction(self.hidden_dim)
        self.decoder = Decoder(self.n_classes, self.hidden_dim)
        # Learned per-class latent anchors: the basins DSM training carves (IRED). Present in both
        # forms so params init identically; only used by the learned energy / ired loss.
        self.anchors = nn.Embed(self.n_classes, self.latent_dim)

    def __call__(self, x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        # Init must touch ALL submodules so their params are created.
        h_x = self.encoder(x)
        e = self.energy(h_x, z)
        _ = self.decoder(z)
        _ = self.anchors(jnp.zeros((1,), dtype=jnp.int32))
        return e

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.encoder(x)

    def center(self, h_x: jnp.ndarray) -> jnp.ndarray:
        return self.center_head(h_x)

    def anchors_all(self) -> jnp.ndarray:
        """``(n_classes, latent_dim)`` learned class anchors (DSM targets)."""
        return self.anchors(jnp.arange(self.n_classes))

    def energy(self, h_x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        if self.energy_form == "learned":
            # Fully-learned scalar field + a mild ridge so descent stays bounded (no fixed bowl,
            # so DSM can carve multiple competing basins -> restart geometry can vary per input).
            corr = self.correction(h_x, z)
            return corr + 0.5 * self.ridge * jnp.sum(z**2, axis=-1)
        c = self.center_head(h_x)
        bowl = 0.5 * jnp.sum((z - c) ** 2, axis=-1)
        corr = CORRECTION_SCALE * jnp.tanh(self.correction(h_x, z))
        return bowl + corr

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)


def build(cfg, n_classes: int, feature_dim: int, key) -> tuple[dict, ReasonerFns, EnergyReasoner]:
    """Initialise params and return ``(params, ReasonerFns, model)``.

    ``cfg`` is an ``edc.config.Config``. ``key`` is a JAX PRNG key from ``edc.seeding``.
    """
    energy_form = "learned" if getattr(cfg.train, "objective", "basin_center") == "ired" else "bowl"
    model = EnergyReasoner(
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        context_dim=cfg.model.context_dim,
        n_classes=n_classes,
        energy_form=energy_form,
        ridge=getattr(cfg.train, "ired_ridge", 0.05),
    )
    dummy_x = jnp.zeros((1, feature_dim))
    dummy_z = jnp.zeros((1, cfg.model.latent_dim))
    params = model.init(key, dummy_x, dummy_z)

    def encode(p, x):
        return model.apply(p, x, method=EnergyReasoner.encode)

    def energy(p, h_x, z):
        return model.apply(p, h_x, z, method=EnergyReasoner.energy)

    def decode(p, z):
        return model.apply(p, z, method=EnergyReasoner.decode)

    fns = ReasonerFns(
        encode=jax.jit(encode),
        energy=energy,          # left un-jitted: differentiated in the inner loop
        decode=jax.jit(decode),
        latent_dim=cfg.model.latent_dim,
        n_classes=n_classes,
    )
    return params, fns, model
