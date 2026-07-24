# Contributing to EDC

## Setup

```bash
uv sync --python 3.12 --extra dev
make test        # must be green before and after any change
```

## Ground rules (see `.claude/CLAUDE.md` for the full invariant list)

- **JAX stays in the core.** Only `energy/`, `inference/`, and `geometry/curvature.py` import
  JAX. Keep the rest framework-agnostic behind `edc.energy.base`.
- **Randomness flows through `edc.seeding`.** No ad-hoc `PRNGKey`. Determinism is tested.
- **Never hand-edit the ledger.** Append results via `edc.ledger.append`.
- **Figures come from the ledger**, via `analysis/make_figures.py`. No one-off plotting.
- **Conformal calibration is disjoint from mapper fitting.** Guarantees assume exchangeability.

## Adding a task family

Implement `edc.tasks.base.Task` (`sample`, `evaluate`, `difficulty`) in a new
`src/edc/tasks/<name>.py`, register it in `edc.registry`, add a `configs/tasks/<name>.toml`,
and provide an **OOD split** (harder/larger instances) — selective prediction is only
interesting in the partially-competent (~70–85% accuracy) regime.

## Adding a geometry feature (Phase 2+)

Add it under `src/edc/geometry/`, surface it from `features.py`, and add a test asserting
its shape and any invariance (e.g. permutation invariance of basin agreement). New features
must be computable under `vmap` over the K restarts.

## Tests

- Offline, CPU-only, fast. Set `JAX_PLATFORMS=cpu`.
- New numerics get a **finite-difference check** (see `tests/test_energy_grad_hvp.py`).
- New conformal code gets an **empirical-coverage** test on synthetic exchangeable data.

## Style

`make fmt` (ruff format + import sort), `make lint`. Line length 100. Prefer readable math:
single-letter `z`, `E`, `h` are fine and expected.
