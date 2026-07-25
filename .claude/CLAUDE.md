# EDC — operating rules & hard invariants

Energy Descent Certificates (EDC): turning the **geometry of an energy-based reasoner's
inference-time descent** into distribution-free reliability certificates. Read
`docs/RESEARCH_PLAN.md` for the science, `docs/SESSION_HANDOFF.md` for current state.

## Hard invariants (do not violate)

1. **Framework:** JAX is the numeric core. It may only appear under `src/edc/energy/*`,
   `src/edc/inference/*`, and `src/edc/geometry/curvature.py`. Everything else is plain
   Python so the core stays swappable behind `edc.energy.base`.
2. **Env:** `uv sync --python 3.12` only. Run code as `PYTHONPATH=src python ...`.
3. **Tests are offline + CPU-only.** No GPU, no network, no model downloads in `tests/`.
   `make test` must stay green. Set `JAX_PLATFORMS=cpu` in test/CI paths.
4. **Determinism:** all randomness flows through `edc.seeding`. Same seed → identical
   trajectory. Never call `jax.random.PRNGKey` ad hoc outside `seeding.py` helpers.
5. **Ledger is append-only.** Never hand-edit `results/ledger.jsonl`; append via
   `edc.ledger`. Every run records its fully-resolved config + git sha + env.
6. **Figures are regenerable.** Every figure/table comes from `analysis/make_figures.py`
   / `make_tables.py` reading the ledger — never from a one-off script or manual step.
7. **Conformal validity:** the nonconformity mapper is fit on a fold **disjoint** from the
   calibration fold. Never calibrate on training data. Guarantees hold only under
   exchangeability — say so wherever a guarantee is claimed.
8. **The falsification test is sacred:** geometry must beat raw terminal energy in ΔAURC.
   Keep the raw-energy baseline wired into every selective-prediction experiment.

## Phase map (see docs/EXPERIMENTS.md)

- Phase 0 skeleton ✅ / Phase 1 base reasoner ✅ / Phase 2 geometry features ✅
- Phase 3 conformal falsification + LTT abstention ✅ (E1: geometry beats energy, ΔAURC CI
  excludes 0; F2/F3 regenerate from the ledger).
- Phase 4a robustness + K-ablation ✅ (`run_sweep` + `aggregate` + `make_tables` T1/T2/T3 + S1
  K-lift figure; S1: geometry lift grows monotonically with restarts, K=1≈EBT barely separates).
- Phase 4b adaptive halting ✅ (opt-in per-step decode + `halting.adaptive` CRC + F4; H1: ~58%
  compute saved at halting risk ≤ α, no accuracy loss). Both guarantees now live.
- Phase 4c more tasks/OOD/F6 · F5 diagnostic · full-fold T1 · Modal · Phase 5 paper — stubbed,
  `NotImplementedError` with a phase note until built.
