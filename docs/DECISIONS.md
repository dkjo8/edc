# Decisions (ADR log)

Short, dated, append-only. Newest first.

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
