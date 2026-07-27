"""Typed, frozen configuration loaded from TOML.

Convention (matching EBRM's ``config.toml`` + Tars's reproducibility discipline):

* configs are TOML; an experiment/task file is deep-merged onto ``configs/base.toml``;
* the *resolved* config is turned into frozen dataclasses for typed access in code;
* the resolved config is also serialisable back to a plain dict so every ledger row
  can record exactly what produced it (see ``edc.ledger``).

Only the keys the code uses are typed here; unknown keys are preserved in ``.extra`` so
task-specific blocks (e.g. ``[task.arithmetic]``) survive round-tripping without needing a
dataclass each.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.toml"


@dataclass(frozen=True)
class RunCfg:
    seed: int = 0
    task: str = "arithmetic"
    notes: str = ""


@dataclass(frozen=True)
class ModelCfg:
    latent_dim: int = 32
    hidden_dim: int = 128
    context_dim: int = 64


@dataclass(frozen=True)
class InferenceCfg:
    k_restarts: int = 8
    steps: int = 50
    step_size: float = 0.1
    temperature: float = 0.05
    init_scale: float = 1.0
    grad_tol: float = 1e-3
    # Inference sampler: "langevin" (fixed-step, default) or "annealed" (simulated-annealing
    # Langevin — a geometric step-size/noise schedule that settles into a mode; needed for the
    # learned IRED landscape). Total steps = anneal_levels * anneal_steps_per_level.
    sampler: str = "langevin"
    anneal_levels: int = 10
    anneal_steps_per_level: int = 8
    anneal_step_max: float = 0.2     # largest per-step size (explore)
    anneal_step_min: float = 0.01    # smallest per-step size (settle into a basin)


@dataclass(frozen=True)
class TrainCfg:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    n_train: int = 4000
    n_neg: int = 4
    neg_noise: float = 0.5
    # Training objective: "basin_center" (Phase-1 supervised bowl, default) or "ired"
    # (denoising score matching toward a learned per-class latent codebook -> a genuinely
    # learned multi-basin landscape; see edc.train.losses.ired_loss_fn).
    objective: str = "basin_center"
    ired_noise_min: float = 0.1      # smallest DSM noise scale sigma
    ired_noise_max: float = 1.5      # largest DSM noise scale sigma (annealing range)
    ired_ridge: float = 0.05         # ||z||^2 ridge keeping the learned energy bounded
    ired_decode_weight: float = 1.0  # weight on the anchor/path decode-CE term
    ired_stat_weight: float = 1.0    # weight on the anchor-stationarity (attractor) term


@dataclass(frozen=True)
class EvalCfg:
    n_eval: int = 1000


@dataclass(frozen=True)
class ConformalCfg:
    alpha: float = 0.1
    delta: float = 0.05
    n_calib: int = 1000


@dataclass(frozen=True)
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    conformal: ConformalCfg = field(default_factory=ConformalCfg)
    # Untyped blocks (e.g. task-specific settings) preserved verbatim.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for hashing / ledger recording (extra folded back in)."""
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return d


_TYPED = {
    "run": RunCfg,
    "model": ModelCfg,
    "inference": InferenceCfg,
    "train": TrainCfg,
    "eval": EvalCfg,
    "conformal": ConformalCfg,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _build(resolved: dict[str, Any]) -> Config:
    typed_kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in resolved.items():
        cls = _TYPED.get(key)
        if cls is not None and isinstance(value, dict):
            known = {f for f in cls.__dataclass_fields__}
            typed_kwargs[key] = cls(**{k: v for k, v in value.items() if k in known})
            leftover = {k: v for k, v in value.items() if k not in known}
            if leftover:
                extra[key] = leftover
        else:
            extra[key] = value
    return Config(**typed_kwargs, extra=extra)


def load_config(path: str | Path | None = None) -> Config:
    """Load ``configs/base.toml`` deep-merged with an optional override file."""
    resolved = _read_toml(BASE_CONFIG)
    if path is not None:
        resolved = _deep_merge(resolved, _read_toml(path))
    return _build(resolved)


def load_from_dict(resolved: dict[str, Any]) -> Config:
    """Build a Config from an already-resolved dict (used when replaying a ledger row)."""
    return _build(resolved)
