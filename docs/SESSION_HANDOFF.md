# Session Handoff

**Start here.** Rolling state of the project between sessions.

## Current state (2026-08-03)

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
- **Phase 4b (adaptive halting — the second guarantee): ✅** opt-in per-step decoding
  (`optimizer.record_z` → `restarts.solve(record_steps=)` → `TrajectoryRecord.step_pred`),
  `halting.adaptive` (basin-agreement policy + CRC calibration via the built `crc.calibrate`,
  `λ=1−τ`), `eval.evaluate_halting`, `experiments/run_halting.py`, F4 in plotting/make_figures.
  **H1 ran:** CRC τ̂=0.917 saves **57.8% compute** at halting risk **0.9% ≤ α=0.1**, no accuracy
  loss, risk monotone in τ. Both guarantees now live (LTT abstention + CRC halting).
- **Phase 4c (arithmetic figure set complete — F5 + F6): ✅** `evaluate` gained `feature_diagnostics`
  (F5) and `ood_validity`/`ood_ltt` (F6, ID-calibrated λ applied to OOD); `plotting.feature_diagnostics`
  + `plotting.ood_stress`; F5/F6 in make_figures. **F6:** at α=0.1 selective risk is 0.075 ID (valid)
  vs **0.762 OOD** — the guarantee breaks under shift, motivating abstention. **F5:** basin-agreement
  features separate correct/incorrect best; curvature weak on arithmetic (reported honestly).
- **Phase 4d (generalization — 2nd task): ✅** `GraphPlanningTask` (BFS shortest-path, larger-graph
  OOD; `tasks/graph_planning.py`), task-aware `analysis.aggregate` + per-task `make_tables` T1,
  `configs/experiments/graph_selective.toml` + `sweeps/graph_seeds.toml`. **E2 (5 seeds):** geometry
  beats energy on **5/5 seeds**, ΔAURC +0.111±0.054 (vs arithmetic +0.083±0.034) — the discovery
  replicates on a structurally distinct family. `plotting.k_restart_lift` made task-aware so graph's
  fixed-K rows don't contaminate the arithmetic S1 figure.
- **Phase 4e (baseline stress test — softmax confidence): ✅ ⚠️ key caveat.** Added MSP,
  temperature-scaled MSP, predictive-entropy baselines (`eval/baselines.py`), a
  `delta_aurc_vs_best_baseline` falsification, task-aware `headline_cell`, per-task T1 with both
  ΔAURCs. **Result (5 seeds/task):** geometry beats scalar **energy** (invariant 8 ✓: +0.085
  arithmetic, +0.111 graph, 5/5 each) but does **not** beat softmax confidence — ΔAURC vs best
  baseline is −0.011 (arith, 0/5) and −0.034 (graph, 0/5). MSP/entropy tie-or-beat geometry (F2).
  **The "geometry is the best nonconformity score" claim fails against softmax confidence** under
  the Phase-1 basin-center reasoner. Likely cause: shallow decoder + simple landscape → softmax
  already informative. **Lever: Phase-4 IRED landscape training** (below).
- **Phase 4f (feature-group ablation — A1/T2): ✅** `evaluate` computes `feature_ablation`
  (leave-one-group-out + group-only mappers); aggregated → `T2b_feature_ablation.tex`. **Result
  (5 seeds/task):** `drop_energy` ≈ `full` and both far below raw energy → geometry's win over
  energy is from the **non-energy** features; **basin agreement is the dominant driver** (dynamics
  second, esp. graph; curvature/energy minor). Rebuts "it's just energy"; consistent with F5.
- **Phase 5 (paper draft): ✅** all seven `paper/sections/*.tex` written (intro, background, method,
  guarantees, experiments, related, conclusion) grounded in the real results; figures F2–F6/S1 and
  tables T1/T2/T2b/T3 wired in; `paper/main.tex` compiles via `tectonic` (`cd paper && tectonic
  main.tex`).
- **Phase 4g (IRED learned landscape — WORKS, flips the 4e caveat): ✅** opt-in `[train]
  objective="ired"`: learned per-class anchor codebook + fully-learned multi-basin energy
  (`mlp_ebm energy_form="learned"`), **contrastive + stationarity + decode** loss
  (`train/losses.ired_loss_fn`) + opt-in **annealed Langevin** sampler
  (`optimizer.annealed_langevin`, `[inference] sampler="annealed"`). The stationarity term
  (`||∇_z E(μ_y)||→0`, weight `ired_stat_weight`) was the key — it makes anchors genuine attractors,
  which the earlier DSM attempts lacked. **5-seed result (task-dependent flip):** on **arithmetic**,
  geometry beats softmax **4/5 seeds** (ΔAURC +0.007±0.005 vs best baseline, vs 0/5 under
  basin_center) — flips the Phase-4e caveat. On **graph** it only **ties**, and this is **not** a
  weak-reasoner artifact: a first IRED graph model (ID~0.58) tied (1/5), and a **stronger one (hidden
  256/latent 48, ID~0.83, matching arithmetic) still ties (0/5, ΔAURC +0.001±0.003)** while beating
  scalar energy 5/5. So the flip is arithmetic-specific — softmax already captures on graph what
  descent geometry would add. `configs/experiments/{arithmetic,graph}_ired.toml`,
  `sweeps/{ired,graph_ired}_seeds.toml`. Aggregation is **objective-aware** (`aggregate.objective_of`,
  `headline_cell(objective=…)`) so IRED never pollutes the basin-center T1 (92 tests green).
- **Phase 4j (complementarity — is the graph tie redundancy or complementarity?): ✅** `evaluate`
  fits a learned-softmax mapper and a `[geometry ∪ softmax]` combined mapper (invariant 7) and reports
  `delta_aurc_geom_adds` (T4). **Answer: redundancy.** Geometry adds conditional signal over softmax
  ONLY where it wins head-to-head (arithmetic-IRED, 4/5, +0.007); on graph-IRED the combined mapper
  ≈ softmax alone (0/5, +0.001) — a well-calibrated softmax already captures what descent geometry
  would add there. `headline_cell` hardened (per-seed timestamp dedup + prefer newest field-carrying
  cell) so re-runs after config-schema changes don't create duplicate/stale headline cells. 95 tests.
- **Phase 4k (richer geometry — is the graph tie thin features or a task property?): ✅** opt-in
  `[eval] richer_geometry=true` appends two groups after the base 14: the **full Hessian spectrum**
  (`geometry/spectrum.py`: lmin, neg-eigenvalue fraction, effective rank, log-det — exact `d×d` eigh)
  and **mode connectivity** (`geometry/connectivity.py`: energy barriers from the best restart to the
  others). New JAX lives in `curvature.py` (`batched_spectrum`, `batched_path_energy`) to respect
  invariant 1; the two feature modules are pure NumPy. Re-ran the 4j complementarity test on all four
  cells (T5). **Answer: the graph redundancy is a task property, not thin features.** Richer geometry
  helps nowhere — graph-IRED conditional signal over softmax stays **−0.002, 0/5**; it's a wash where
  geometry already wins (arith-IRED +0.007, 4/5, unchanged) and slightly worse on the fixed-bowl
  basin-center cells (extra features add noise). Aggregation is feature-set-aware (`feature_set_of`;
  `headline_cell`/`by_k` default `feature_set="base"`) so richer never pollutes T1/T2/T4. 109 tests.
- **Phase 4l (certificates on firm footing — multi-seed guarantees + graph certification): ✅** put
  the project's core contribution (the certificates) on a 5-seed footing. New
  `experiments/run_halting_sweep.py` + `aggregate.{halting_rows, aggregate_halting_cell}` → **T6**:
  arith halting saves **57.1%±1.7%** (was single-seed 57.8%), graph **80.8%±5.2%**, both halting risk
  ≤ α, within budget 5/5. `[sweep] include_ood=true` (run_sweep) + an OOD aggregation block → **T7**:
  the selective guarantee breaks under shift over 5 seeds (arith ID sel-risk 0.077 → OOD 0.764, 0/5;
  graph 0.010 → 0.168, 4/5). Full-fold graph cert run (`graph_cert_seeds.toml`, n_calib=1500): LTT
  certifies only **1.1% coverage at α=0.1** but **22.5% at α=0.15, 48.5% at α=0.20; validity holds at
  every α** — the old "certifies zero" was an operating-point artifact of graph's ~30% error floor,
  not a broken certificate. Fixed a stale-row bug (halting dedups by seed, dropping the Phase-4b H1
  row under a drifted config_hash) and made `plotting._selective_row`/`_halting_row` prefer arithmetic
  so the new full-fold (n_test=1500) graph rows don't flip F2/F4/F6. 115 tests.
- **Phase 4m (deep-ensemble baseline — the last deferred baseline): ✅** opt-in `[eval] ensemble_size`
  trains M=5 independent models (member 0 = primary; `baselines.ensemble_scores` = mean-softmax
  MSP/entropy + BALD mutual-information; `seeding.member_seeds` from a fresh substream; all members
  run on identical fold inputs). Reported **separately** (`delta_aurc_vs_best_ensemble`, T8), not
  folded into the same-compute T1, because it's M× compute. **Result:** geometry beats the 5-model
  ensemble on selective risk **only on arith-IRED (5/5, ΔAURC +0.017) — despite the ensemble being
  more accurate (0.92 vs 0.84)**; elsewhere the ensemble wins/ties (the same cells where geometry
  lost/tied softmax). Mirrors the softmax finding; rebuts "geometry is just cheap ensembling" exactly
  where the positive claim lives. Fixed a `headline_cell` grouping bug (OOD and ensemble cells sharing
  (K, n_test) merged and deduped the OOD rows away — now group by field-presence + a `prefer` bias;
  `plotting._selective_row`/`_halting_row`/`ood_stress` prefer arithmetic / OOD-carrying rows). 128 tests.
- **Open question / next:** a 3rd/4th task (E3/E4, e.g. logic/parity) to characterize *when* geometry
  beats softmax now that feature poverty is ruled out; MC-dropout (the one still-deferred baseline);
  Modal; scale-up.

## What is real vs stub

- Real: config/seeding/ledger, arithmetic task (+`ood_max_operand` shift) + **graph shortest-path
  task** (size-shift OOD), energy/encoder/decoder, Langevin optimizer+restarts (opt-in per-step
  decode), training loop,
  curvature HVP/λ_max/trace + batched wrapper, **all geometry features + assembly**,
  `single_feature_auroc`, `edc.cli geometry`; **the full selective-prediction stack**: AURC/ΔAURC,
  nonconformity mapper, split-conformal/LTT/CRC, `eval.evaluate`, `run_experiment.py`, F2/F3;
  **the sweep/aggregation/tables stack**: `run_sweep.py`, `analysis.aggregate`, `make_tables`
  (T1–T3), S1 K-lift figure; **adaptive halting**: `halting.adaptive`, `eval.evaluate_halting`,
  `run_halting.py`, F4; **F5 mechanism + F6 OOD stress** (`feature_diagnostics`, `ood_validity`);
  **softmax-confidence baselines** (`eval/baselines.py`: MSP/temp/entropy); **richer geometry**
  (opt-in `geometry/spectrum.py` full-spectrum + `geometry/connectivity.py` mode-connectivity, with
  `curvature.batched_spectrum`/`batched_path_energy` in the JAX core; T5).
- Stub: **IRED landscape training** (`train/losses.py` is still the Phase-1 basin-center scheme),
  logic/hard tasks (E3/E4), Modal runner, paper write-up.

## Next steps (Phase 4f) — reoriented by the 4e caveat

1. **IRED landscape training (the priority).** Replace the supervised basin-center loss
   (`train/losses.py`) with an IRED-style annealed-landscape / denoising-score-matching objective so
   the energy is a genuinely learned multi-basin surface. Hypothesis: restart geometry then carries
   signal the (now less-informative) softmax does not. **Re-run the 4e baseline stress test** — this
   is the make-or-break experiment for the discovery's practical claim.
2. **If geometry still ties softmax:** reframe honestly — the contribution is the distribution-free
   *certificate* machinery (LTT/CRC + F2–F6) over an EBM reasoner, with geometry beating the EBT
   scalar-energy signal specifically (invariant 8), not a universal win over all confidences.

3. **Lower priority (after the IRED verdict):** a 3rd task (logic/sudoku, E3/E4) for more
   generalization; full-fold ≥5-seed E1 for a headline T1; paper write-up (figure set F2–F6/S1 and
   tables T1–T3 are complete for arithmetic + graph).

## Operating regime note

E1 is tuned to ~77–81% ID base accuracy via `n_operands=6, max_operand=7, epochs=7` (base error >
α, so abstention is genuinely exercised). The smoke config saturates (~97%) and its `n_calib=128` is
too small for LTT to certify — that is expected, not a bug; use the E1 config for real numbers.

## Invariants to keep green

`make test` (offline, CPU), determinism, ledger append-only, JAX confined to core, raw-energy
baseline wired into every selective experiment. See `.claude/CLAUDE.md`.
