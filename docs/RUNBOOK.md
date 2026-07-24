# Runbook

## Local (CPU, offline)

```bash
uv sync --python 3.12 --extra dev     # pinned env
make test                             # offline CPU test suite
make smoke                            # end-to-end base reasoner (<60s) -> ledger row
make lint                             # ruff
```

Force CPU for JAX if a GPU/plugin is present:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src python -m edc.cli smoke --config configs/smoke.toml
```

## GPU (Modal, Phase 4)

```bash
# Larger experiments/sweeps on a Modal GPU box (image pins jax[cuda12]).
PYTHONPATH=src python experiments/modal_runner.py    # (Phase 4)
make sweep CONFIG=configs/sweeps/k_restarts.toml     # (Phase 4)
```

Results sync back to `results/ledger.jsonl`; then `make figures && make tables`.

## Common issues

- **JAX picks the wrong backend** → set `JAX_PLATFORMS=cpu`.
- **Smoke accuracy at chance** → check `τ>0`, that training loss is decreasing, and that the
  codebook scale matches `init_scale` (descent must be able to reach anchors).
- **Non-deterministic run** → some randomness bypassed `edc.seeding`; all keys must derive from
  `root_key(seed)` / `numpy_rng(seed, ...)`.
- **Ledger edited by hand** → don't; append via `edc.ledger.append` only.
