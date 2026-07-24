# Session Handoff

**Start here.** Rolling state of the project between sessions.

## Current state (2026-07-24)

- **Phase 0 (skeleton): ✅** repo tree, `uv`/`pyproject.toml`, `Makefile`, CI, config system
  (`edc.config`), deterministic seeding (`edc.seeding`), append-only ledger (`edc.ledger`),
  registry, CLI, all docs, paper skeleton.
- **Phase 1 (base reasoner): ✅** JAX/Flax energy reasoner (`edc.energy.mlp_ebm`), K-restart
  Langevin inference (`edc.inference`), IREM-style contrastive training (`edc.train`),
  arithmetic task (`edc.tasks.arithmetic`). `make smoke` runs end-to-end on CPU and appends a
  ledger row. HVP curvature primitive (`edc.geometry.curvature`) implemented + tested early
  because it is the method's crux.
- **Phases 2–5: stubbed** with `NotImplementedError` + phase notes.

## What is real vs stub

- Real: config/seeding/ledger, arithmetic task, energy/encoder/decoder, Langevin
  optimizer+restarts, training loop, curvature HVP/λ_max/trace, split-conformal threshold.
- Stub: geometry `features/basin/energy_stats/dynamics`, conformal `ltt/crc/selective/
  nonconformity`, halting, eval, graph/logic/hard tasks, experiment/sweep/modal runners,
  figure/table generators.

## Next steps (Phase 2)

1. Implement `edc.geometry.{basin,energy_stats,dynamics}` from `TrajectoryRecord`; assemble in
   `features.geometry_features` (must run under `vmap` over restarts).
2. Add `tests/test_geometry_features.py` (shapes, invariances) + an F5-style diagnostic showing
   geometry separates correct vs incorrect on arithmetic.
3. Then Phase 3: nonconformity mapper → LTT/CRC → selective/halting, with the coverage-validity
   test and the ΔAURC falsification harness.

## Invariants to keep green

`make test` (offline, CPU), determinism, ledger append-only, JAX confined to core, raw-energy
baseline wired into every selective experiment. See `.claude/CLAUDE.md`.
