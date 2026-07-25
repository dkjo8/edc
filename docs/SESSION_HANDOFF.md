# Session Handoff

**Start here.** Rolling state of the project between sessions.

## Current state (2026-07-25)

- **Phase 0 (skeleton): ✅** repo tree, `uv`/`pyproject.toml`, `Makefile`, CI, config system
  (`edc.config`), deterministic seeding (`edc.seeding`), append-only ledger (`edc.ledger`),
  registry, CLI, all docs, paper skeleton.
- **Phase 1 (base reasoner): ✅** JAX/Flax energy reasoner (`edc.energy.mlp_ebm`), K-restart
  Langevin inference (`edc.inference`), IREM-style contrastive training (`edc.train`),
  arithmetic task (`edc.tasks.arithmetic`). `make smoke` runs end-to-end on CPU and appends a
  ledger row. HVP curvature primitive (`edc.geometry.curvature`) implemented + tested early
  because it is the method's crux.
- **Phase 2 (geometry features): ✅** `edc.geometry.{basin,energy_stats,dynamics}` (plain
  NumPy, invariant 1) + `features.geometry_features` assembling a 14-feature per-input vector
  (basin 3 · energy 3 · curvature 4 · dynamics 4). Curvature is batched over particles by the
  new `curvature.batched_curvature` (the only JAX in the feature path). `TrajectoryRecord` now
  carries `h_x` so curvature can be recomputed at `z*`. `edc.cli geometry` prints per-feature
  correct-vs-incorrect AUROC (geometry vs the raw-energy baseline) and appends a ledger row.
- **Phase 3 (conformal + falsification + abstention): ✅** `eval.metrics` (`risk_coverage_curve`,
  tie-robust `aurc`, `paired_bootstrap_delta_aurc`, `selective_accuracy_at_coverage`, `ece`);
  `conformal.nonconformity` (sklearn logistic mapper + `1−ρ_basin` fallback), `conformal.selective`,
  `conformal.ltt` (Hoeffding-Bentkus p-value + Bonferroni LTT), `conformal.crc.calibrate`;
  `eval.evaluate` (disjoint fit/calib/test folds) + `experiments/run_experiment.py`; F2/F3 in
  `plotting` + `analysis/make_figures.py`. **E1 ran: geometry beats raw energy** (ΔAURC +0.074, 95%
  CI [+0.054, +0.093]); LTT holds (69% coverage @ selective risk 0.075 ≤ α=0.1). See EXPERIMENTS.md.
- **Phase 4a (robustness + K-ablation + reporting): ✅** `experiments/run_sweep.py` (grid+override,
  reuses `run_experiment.run_and_append`; `evaluate(include_ood=False)` for cheaper cells),
  `analysis/aggregate.py` (multi-seed ledger aggregation), `analysis/make_tables.py` (T1/T2/T3 →
  `paper/tables/*.tex`), and the S1 K-lift figure. **S1 ran (5 seeds × K∈{1,2,4,8,16}):** the ΔAURC
  lift grows monotonically with K — K=1 (≈EBT) separates on only 1/5 seeds, K≥8 on 5/5. The
  restart geometry is the mechanism; the "K>1 gives no lift" falsifier does not fire. See EXPERIMENTS.md.
- **Deferred / Phases 4b–5: stubbed.** Adaptive halting (`halting.adaptive`) + F4, F5 feature-
  distribution diagnostic, graph/logic/hard tasks + F6, Modal runner, ≥5-seed full-fold E1 for T1.

## What is real vs stub

- Real: config/seeding/ledger, arithmetic task (+`ood_max_operand` magnitude shift),
  energy/encoder/decoder, Langevin optimizer+restarts, training loop, curvature HVP/λ_max/trace +
  batched wrapper, **all geometry features + assembly**, `single_feature_auroc`, `edc.cli geometry`;
  **the full selective-prediction stack**: AURC/ΔAURC, nonconformity mapper, split-conformal/LTT/CRC,
  `eval.evaluate`, `run_experiment.py`, F2/F3.
  `eval.evaluate`, `run_experiment.py`, F2/F3; **the sweep/aggregation/tables stack**:
  `run_sweep.py`, `analysis.aggregate`, `make_tables` (T1–T3), S1 K-lift figure.
- Stub: `halting.adaptive` + F4, F5 feature-distribution diagnostic, graph/logic/hard tasks,
  Modal runner.

## Next steps (Phase 4b)

1. **Adaptive halting (F4):** add per-step decoded answers to `TrajectoryRecord` (optimizer emits
   `z` per scan step → decode per step), implement `halting.adaptive.halting_policy` consuming the
   already-built + tested `crc.calibrate`, emit the compute-vs-accuracy Pareto.
2. **Full-fold ≥5-seed E1 for T1:** the sweep's T1 uses reduced folds (n_test=600) at K=16; add a
   seed sweep at the full E1 config (n_test=1500) for the headline table.
3. **More tasks + OOD stress (F6):** graph planning / logic / hard sudoku; the magnitude-shift OOD
   (`ood_max_operand`) is wired — extend `evaluate` to report the guarantee *breaking* under shift.
4. **F5 mechanism diagnostic:** store per-feature correct-vs-incorrect summaries in `evaluate` and
   plot basin/curvature distributions.

## Operating regime note

E1 is tuned to ~77–81% ID base accuracy via `n_operands=6, max_operand=7, epochs=7` (base error >
α, so abstention is genuinely exercised). The smoke config saturates (~97%) and its `n_calib=128` is
too small for LTT to certify — that is expected, not a bug; use the E1 config for real numbers.

## Invariants to keep green

`make test` (offline, CPU), determinism, ledger append-only, JAX confined to core, raw-energy
baseline wired into every selective experiment. See `.claude/CLAUDE.md`.
