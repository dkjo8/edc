# Decisions (ADR log)

Short, dated, append-only. Newest first.

## 2026-07-27 — Phase 4h: objective-aware aggregation; the IRED flip is task-dependent
**Decision:** `evaluate` now records `objective` (basin_center|ired) + `sampler` in its metrics, and
`analysis/aggregate` is **objective-aware** (`objective_of`, `selective_rows(objective=…)`,
`headline_cell(objective="basin_center")` by default) so IRED rows never contaminate the canonical
basin-center T1 (previously the IRED and basin sweeps shared K=12 and T1 picked one by a fragile
tie-break). **Finding:** the Phase-4g softmax-caveat flip is **task-dependent** — robust on
arithmetic (geometry beats softmax 4/5 under IRED) but only a tie on graph (1 win/1 loss/3 ties),
where the learned reasoner is weaker (ID 0.55–0.82). Reported narrowly; the paper/abstract/docs
tempered accordingly. **Why:** honesty — one task's flip is not a general claim; and the fragile
tie-break was a latent T1 bug worth fixing. **Reversible?** Yes — `objective` filter is additive.

## 2026-07-27 — Phase 4g: IRED reasons via contrastive+stationarity (not DSM); flips the 4e caveat
**Decision:** the working IRED objective is **IREM-style contrastive + stationarity + decode**, not
denoising score matching. DSM was stuck (score net never fit under the double-grad objective).
Contrastive alone (anchor below competitors/random) makes the correct anchor *low-energy* but not a
*local minimum*, so descent never settles there; the fix is a **stationarity** term
`||∇_z E(μ_y)||²→0` (weight `ired_stat_weight`) that makes anchors genuine attractors. With the
annealed Langevin sampler this reasons (ID 0.72–0.90), and under the learned landscape geometry beats
softmax confidence 4/5 seeds — reversing the Phase-4e caveat. **Why it matters:** it isolates the
cause of that caveat (the simple basin-center landscape, not the geometry idea). Still opt-in
(`objective="ired"`, default `basin_center` unchanged) and additive; merged to `main` as a working,
tested option. IRED ledger rows are a distinct `(K, n_test)`/config group so they do not pollute the
basin-center T1 aggregates.

## 2026-07-27 — Phase 4g: IRED training is opt-in and its energy/anchors are additive (WIP)
**Decision:** the IRED landscape objective is an **opt-in** `[train] objective="ired"` that switches
`mlp_ebm` to a fully-learned `energy_form="learned"` (no bowl) + a per-class anchor `nn.Embed`, and
uses `train.losses.ired_loss_fn` (denoising score matching toward anchors + reachability + decode).
The default stays `basin_center`/`bowl`, so every existing result/test is untouched. It is developed
on a branch and **not merged** into `main`/v0.0.1. **Why:** DSM/EBM training is high-risk; keeping
it additive means the release stays clean regardless of outcome, and the honest WIP status (trains
but does not yet reason under fixed-step Langevin — needs annealed sampling) is recorded rather than
forced. **Reversible?** Fully — deleting the branch / leaving the objective unused changes nothing.

## 2026-07-27 — Phase 4f: feature-group leave-one-out ablation (which geometry features drive it)
**Decision:** `evaluate` computes a `feature_ablation` block — refit the logistic geometry mapper on
feature subsets (leave-one-group-out and group-only across basin/energy/curvature/dynamics) on the
fit fold and score the test fold (`_feature_ablation`). Aggregated over seeds and emitted as
`T2b_feature_ablation.tex`. **Why:** the geometry vector *includes* the 3 energy statistics, so
"geometry beats energy" could be partly re-using energy; `drop_energy` (geometry without its energy
columns) vs `full` isolates the genuine basin/curvature/dynamics contribution — directly probing the
Phase-4e caveat. Cheap (logistic on ≤14 features), no training-recipe risk. **Reversible?** Yes —
additive metric block; older rows aggregate to an empty ablation.
**Note:** full IRED annealed-landscape / score-matching training (the priority lever from 4e) is a
substantial research effort scoped as the next phase, not folded into the v0.0.1 release; the
release ships the complete certificate machinery + honest findings under the Phase-1 reasoner.

## 2026-07-27 — Phase 4e: softmax-confidence baselines + a best-of-all-baselines falsification
**Decision:** add three standard softmax-confidence nonconformity baselines (`eval/baselines.py`) —
**MSP**, **temperature-scaled MSP**, and **predictive entropy** — computed from the mean decoder
logits over the K restarts. Temperature is fit on the **fit** fold only (invariant 7). `evaluate`
now reports AURC for these alongside the energy baselines, and adds
`delta_aurc_vs_best_baseline` (geometry vs the strongest of *all* baselines) with a bootstrap CI,
**in addition to** the sacred `delta_aurc_vs_best_energy` (invariant 8). T1 shows both ΔAURCs per
task. **Why:** "geometry beats energy" is vulnerable to "energy is a weak baseline"; a paper needs
geometry to beat a standard softmax confidence too. Mean-logit read-off gives one temperature-
scalable confidence vector per input; these are cheap (decoder logits already exist). **Deferred:**
MC-dropout and deep ensembles (need model/dropout changes or multiple trainings; the K-restart
best-of-N already supplies ensemble-like diversity). **Honesty:** the run records the verdict vs the
best baseline whatever it is; a tie with MSP would be reported, not hidden. **Reversible?** Yes —
additive scores + metric fields; aggregation falls back to the energy ΔAURC for pre-4e rows.

## 2026-07-26 — Phase 4d: second task (graph shortest-path) + task-aware aggregation
**Decision:** the generalization task is **graph shortest-path length** (`tasks/graph_planning.py`):
random Erdős–Rényi graph, random `(source, target)`, label = BFS shortest-path length capped at
`max_len` (class 0 = unreachable/farther). Fixed `feature_dim` via `max_nodes` padding (flattened
adjacency ⊕ source/target one-hots) so one model spans the ID split (`n_nodes`) and the
size-generalization OOD (`ood_n_nodes` > it). Tuned to ID ≈ 0.72 (majority-class baseline ≈ 0.41 →
real learning) via `n_nodes=7, edge_prob=0.4, max_len=4, epochs=25`. `analysis/aggregate.py` gains a
`task` filter (and `tasks_present`) so multi-task ledgers do not cross-contaminate per-K aggregates;
`make_tables` T1 is now one row per task. **Why:** a relational/combinatorial family maximally
distinct from additive arithmetic is the strongest generalization test; edge density is the cleanest
difficulty knob (denser → shorter, more learnable paths); larger graphs are a natural covariate
shift. The whole E1 pipeline is task-agnostic, so the task rides `evaluate`/`run_sweep` unchanged.
**Reversible?** Yes — additive task + optional aggregation arg; arithmetic results untouched.

## 2026-07-25 — Phase 4c: F6 exhibits the guarantee breaking OOD; F5 stores compact diagnostics
**Decision:** F6 (OOD stress) is produced by calibrating the LTT selective-risk threshold on the
**ID** calibration fold and then evaluating the achieved risk on the **OOD** test fold across the
α-sweep (`evaluate` gains `ood_validity` + an `ood_ltt` block, guarded by `include_ood`). F5 stores
`feature_diagnostics` = per-feature `single_feature_auroc` + 20-bin correct/incorrect histograms
(~850 numbers), **not** raw per-input features. **Why:** the exchangeability that the LTT/CRC
guarantees assume is exactly what a distribution shift violates, so applying the *ID-calibrated*
threshold to OOD is the honest way to show the guarantee failing (the argument for abstention in
critical systems); storing histograms keeps the figure regenerable from the ledger (invariant 6)
without bloating it. **Honesty:** F6 is a negative result by construction — the row records whatever
happened (`risk_within_budget`), it is never forced. **Reversible?** Yes — both are additive metric
blocks; `include_ood=False` (the sweep path) omits F6.

## 2026-07-25 — Phase 4b: adaptive halting — opt-in per-step decodes, basin-agreement signal, λ=1−τ
**Decision:** Adaptive halting stops the descent when **basin agreement** (plurality fraction of
decoded answers across restarts) at a step crosses a threshold τ; the halting loss
`L = 1[early best-of-N ≠ full-budget best-of-N]` is CRC-calibrated via the existing
`conformal.crc.calibrate`. Three implementation choices: (1) per-step decoding is **opt-in**
(`langevin_descent(record_z=…)` → `restarts.solve(record_steps=…)` → optional
`TrajectoryRecord.step_pred`), so E1/sweep pay nothing and the descent is byte-identical with it off;
(2) the halting signal is **basin agreement** — the method's own geometry signal, not an ad-hoc
grad-norm proxy; (3) `halting.adaptive.calibrate` reparametrises **`lambda = 1 − tau`** so ascending
λ = lower agreement bar = stop earlier = weakly more errors + more compute saved, matching
`crc.calibrate`'s "largest admissible λ = most compute saved", and returns `tau_hat = 1 − lambda_hat`.
**Why:** confines JAX to the core (invariant 1), keeps the common paths cheap, and reuses the
tested CRC calibrator unchanged. **Monotonicity caveat:** CRC assumes risk monotone in the
threshold; `L(τ)` is monotone only in the limit (answers can flip), so `evaluate_halting` reports
the risk-vs-τ curve + a `risk_monotone_in_tau` flag, and `crc.calibrate` scans conservatively (no
cherry-picking dips). If materially non-monotone, LTT (already built) is the rigorous fallback —
we do not silently claim CRC validity. **Reversible?** Yes — `record_steps` defaults off; the
signal/threshold live behind `halting.adaptive`.

## 2026-07-25 — Phase 4a: sweep harness (override+grid), reduced folds, multi-seed ΔAURC aggregation
**Decision:** `experiments/run_sweep.py` expands a `[sweep.grid]` of dotted keys (cartesian) over a
`base` config, with optional constant `[sweep.override]`; every cell reuses `run_experiment.run_and_append`
(the single train→evaluate→ledger path). The S1 K-sweep runs with **reduced folds**
(`eval.n_eval=600`, `conformal.n_calib=800`) and **`evaluate(include_ood=False)`** (3 folds not 4) to
keep 25 cells CPU-tractable. Multi-seed ΔAURC is aggregated in `analysis/aggregate.py` from each
row's **own point estimate + 95% CI** (mean±std over seeds; count of seeds whose CI excludes 0) —
**not** by pooling raw scores across seeds, which would break per-run exchangeability. **Why:** the
falsification metric (ΔAURC) is threshold-free and stable at n_test=600, so smaller folds are fine
for the ablation; each seed already carries a valid within-run bootstrap CI, so the honest
cross-seed summary is "how many seeds independently clear 0," not a re-bootstrap. **Reversible?**
Yes — folds/grid are config; `include_ood` defaults to True (E1 unchanged); K=32 can be re-added.

## 2026-07-24 — Phase 3: LTT uses Bonferroni over a bounded grid, not ascending fixed-sequence
**Decision:** `conformal.ltt.calibrate` controls FWER with **Bonferroni** (`p(lambda) <= delta/m`)
over a bounded quantile grid of `m` candidate thresholds, and picks the max-coverage admissible
`lambda_hat`. **Why:** the selective risk `R_sel(lambda)` is non-monotone, and a fixed-sequence on
*ascending* lambda (the initial design) stalls immediately — the smallest thresholds answer ~1
point, so Hoeffding-Bentkus has no power and the very first hypothesis fails. Bonferroni tests all
thresholds, so it finds the mid-range one that controls risk even when base error > alpha (the
regime where selective prediction is useful). Validity is confirmed by
`tests/test_conformal_ltt.py::test_ltt_coverage_guarantee` (0 violations over 250 draws).
**Reversible?** Yes — pass explicit `lambdas` or swap the correction behind the same signature.

## 2026-07-24 — AURC is tie-robust (group-mean errors)
**Decision:** `eval.metrics` sorts by score and replaces each point's error with its **tie-group
mean** (expected risk under random tie-breaking) before sweeping coverage. **Why:** scores with few
distinct values (`rho_basin` has only K+1 levels) otherwise give an input-order-dependent AURC,
biasing the geometry-vs-energy comparison. Group-mean errors make AURC order-independent and make
"constant score => AURC = base error rate" exact. Standard split-conformal thresholds admit whole
tie-groups anyway, so the calibration layer is unaffected.

## 2026-07-24 — 70-85% regime via plain-sum operand count; graceful OOD via a magnitude shift
**Decision:** E1 (`configs/experiments/arithmetic_selective.toml`) runs plain-sum arithmetic with
`n_operands=6, max_operand=7` and `epochs=7`, landing ID base accuracy ~77-81% (base error > alpha,
so abstention is genuinely exercised). OOD is a **magnitude covariate shift** (`ood_max_operand=9`,
same operand count) — a new `ArithmeticTask.ood_max_operand` knob. **Why:** the modular "grokking"
variant is bimodal (saturates or collapses to chance) and its operand-count OOD is always at chance;
plain-sum accuracy is smoothly tunable via epochs, and a magnitude shift keeps the low-sum region
in-distribution so OOD degrades *gracefully* (~20% acc, 15x chance) instead of the old degenerate
0.8% (5-operand sums hit label classes never trained). Feature/class dims cover both splits so one
model handles ID+OOD. **Reversible?** Yes — `ood_max_operand` defaults to `max_operand` (no shift).

## 2026-07-24 — Phase 3 scope: falsification + abstention now; adaptive halting deferred
**Decision:** this pass built the selective-prediction falsification (AURC/ΔAURC + geometry mapper
+ LTT abstention + `evaluate`/`run_experiment` + F2/F3) and the pure `crc.calibrate`. The **halting
policy** (`halting.adaptive`) and figure **F4** are deferred. **Why:** halting needs per-step decoded
answers on `TrajectoryRecord` (a schema change), and the discovery claim — geometry beats scalar
energy — does not depend on it. `crc.calibrate` is implemented and tested so the conformal layer is
complete; only the trajectory-coupled policy waits. **Reversible?** Yes — additive when built.

## 2026-07-24 — Geometry features are plain NumPy; curvature batching stays in the JAX core
**Decision:** `geometry/{basin,energy_stats,dynamics,features}.py` are pure NumPy over
`TrajectoryRecord` arrays. All JAX for the features — the per-particle HVPs and their `vmap`
over restarts — lives in a new `curvature.batched_curvature`, whose output `features.py`
converts straight back to NumPy.
**Why:** invariant 1 confines JAX to `energy/*`, `inference/*`, and `geometry/curvature.py`.
The stub docstrings' "computable under vmap over restarts" is met by fixed-shape vectorized
NumPy over the K axis (and real `jax.vmap` inside `curvature.py`).
**Reversible?** Yes — the feature functions take arrays, so a JAX rewrite behind the same
signatures is local if ever needed.

## 2026-07-24 — `h_x` is stored on `TrajectoryRecord`
**Decision:** add `h_x (B, context_dim)` to `TrajectoryRecord`, populated in
`inference.restarts.solve`. **Why:** curvature at `z*` needs the per-input context, but
`geometry_features(traj, fns, params, key)` receives no `x`; storing the already-computed
`h_x` keeps that signature and avoids re-encoding. It is genuine raw material, consistent with
the record keeping the full descent rather than just the endpoint.

## 2026-07-24 — JAX as the numeric core
**Decision:** JAX/Flax for `energy/`, `inference/`, `geometry/curvature.py`; everything else
plain Python behind `edc.energy.base.ReasonerFns`.
**Why:** the crux workload is `vmap` over K restarts + per-input Hessian-vector products
(`jvp∘grad`) + a `lax.scan` inner loop + gradient-through-optimization. JAX composes these
natively with far less friction than `torch.func`, and its deterministic `PRNGKey`/`fold_in`
makes "regenerate figure from seed" exact. No PyTorch EBRM checkpoint exists to reuse (EBRM is
Julia), so nothing is lost by not matching Tars's PyTorch stack.
**Reversible?** Yes — JAX is isolated to three module areas behind one interface.

## 2026-07-24 — Selective vs halting use different conformal tools
**Decision:** Conformal Risk Control for halting; Learn-then-Test for selective/abstention.
**Why:** halting risk is **monotone** in one threshold (CRC's assumption); selective risk
`P(err|answered)` is **non-monotone**, which CRC cannot handle but LTT can (hypothesis testing
with valid p-values). Using the wrong tool would silently void the guarantee.

## 2026-07-24 — Phase-1 training uses a fixed per-class codebook anchor
**Decision:** shape the landscape with contrastive + decode losses against a frozen random
per-class latent anchor, rather than the full IRED annealed-landscape + score-matching recipe.
**Why:** minimal, fast (<60s CPU smoke), and genuinely produces basins to analyse in Phase 2.
**Follow-up:** upgrade to IRED-style training in Phase 4 (tracked in `RESEARCH_PLAN.md`).

## 2026-07-24 — Sticky append-only JSONL ledger is the single source of truth
**Decision:** all results land in `results/ledger.jsonl` via `edc.ledger`; figures/tables read
only from it. **Why:** matches Tars conventions; makes every paper number traceable to a
`(git_sha, config_hash, seed)`.
