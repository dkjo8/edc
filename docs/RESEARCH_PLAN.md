# Research Plan — Energy Descent Certificates (EDC)

## Thesis

An energy-based reasoner answers by **optimizing** a learned energy `E(h_x, z)` over a latent
`z` — it descends to an answer. We claim the **geometry of that descent is a calibratable
predictor of correctness**, and that distribution-free risk control turns it into a
*certificate*: a guaranteed error rate among answered inputs (abstention) and a guaranteed
error budget under early stopping (adaptive halting).

This is the "when to trust" layer that Polished Snow's products currently outsource to an
external verifier (Lean in Tars, the CPU oracle in Kipp). EBRM contributes *how to reason*;
EDC contributes *when to trust the reasoning*.

## The discovery claim (and how it is falsified)

> The per-input geometry of the inference-time energy landscape over the latent —
> **basin agreement across K stochastic restarts, local curvature at the optimum, and descent
> dynamics** — beats the scalar terminal energy as a nonconformity score, and yields a
> distribution-free selective-prediction guarantee for an EBM reasoner.

**Falsification.** Across tasks, if the paired-bootstrap 95% CI on
`ΔAURC(raw terminal energy − geometry)` includes 0, the discovery fails: geometry adds nothing
over the scalar energy that Energy-Based Transformers already use. Secondary falsifiers: `K>1`
gives no lift over `K=1`; curvature/basin features have negligible importance beyond energy;
the conformal guarantee fails to hold in-distribution.

## Base reasoner (reused, not the contribution)

`x → f_φ → h_x`; energy `E_θ(h_x, z)`; decoder `g_ψ: z* → ŷ`. Inference is **K stochastic-restart
Langevin descent**

```
z_{t+1} = z_t − η ∇_z E_θ(h_x, z_t) + √(2 η τ) ε_t,   ε_t ~ N(0, I),   τ > 0.
```

`τ > 0` is required — deterministic descent collapses restarts into one basin and destroys the
basin-agreement signal. Training (Phase 1) shapes the landscape with a contrastive objective so
descent lands `z` in the basin that decodes to the correct answer (IREM/IRED-style); the
Phase-1 implementation uses a fixed per-class codebook anchor with contrastive + decode losses
(`src/edc/train/losses.py`). Phase 4 upgrades this to the full IRED annealed-landscape +
score-matching recipe.

## Geometry features (the contribution) — Phase 2

Per input, from the `K`-restart `TrajectoryRecord`:

1. **Basin agreement** — plurality fraction `ρ_basin` of decoded answers across restarts,
   decoded-answer entropy, latent-cluster separation. *High agreement ⇒ reliable.*
2. **Terminal-energy statistics** — mean, min, spread of `E(z*)` across restarts. *These are the
   EBT scalar-energy baseline; here they are just features to beat.*
3. **Local curvature at `z*`** — Hessian of the energy **over the latent**, per input, via
   Hessian-vector products (`src/edc/geometry/curvature.py`):
   - `λ_max` (sharpness) by power iteration on `Hv = jvp(∇_z E, z, v)`;
   - `tr(H)` (basin volume proxy) by a Hutchinson estimator.
   *Sharp basin ⇒ hypothesised less reliable; wide/flat ⇒ confident.* This is a **different
   object** from the loss-over-weights curvature studied in the calibration/sharpness
   literature (`2604.20614`) — that protects the novelty.
4. **Descent dynamics** — steps-to-convergence, energy-decrease rate, monotonicity fraction,
   terminal gradient norm (residual stationarity). *Slow / non-monotone / high-residual ⇒
   unreliable.*

**Nonconformity score.** A small monotone mapper (logistic / GBT) maps the feature vector to
`p̂(correct)`, fit on a fold **disjoint** from the calibration fold (conformal validity). Score
`s(x) = 1 − p̂(correct)`. A hand-defined `1 − ρ_basin` variant is kept so validity does not hinge
on the learned mapper.

## Two guarantees, two tools — Phase 3

- **Adaptive halting → Conformal Risk Control** (`2208.02814`). The halting loss
  `L(λ) = 1[early-stopped answer ≠ full-budget answer]` is bounded and **monotone** in a single
  threshold `λ`, so CRC picks `λ̂` with `E[L(λ̂)] ≤ α` in finite samples.
- **Selective prediction / abstention → Learn-then-Test** (`2110.01052`). The selective risk
  `R_sel(λ) = P(error | answered)` is **non-monotone**, so LTT treats each `λ` as a hypothesis,
  uses a valid Hoeffding–Bentkus p-value, and returns the FWER-controlled admissible set:
  `P(R_sel(λ̂) ≤ α) ≥ 1 − δ`, distribution-free, under exchangeability.

Both guarantees are **marginal over the calibration draw and assume exchangeability**; we
deliberately show them *break* under distribution shift (OOD split), which is the argument for
abstention/routing in critical systems.

## Experiments — Phase 4

Tasks: arithmetic (Phase 1), graph planning, logic (ported from EBRM), and hard 9×9 Sudoku
(IRED) — each with in-distribution + OOD splits, operated at the **70–85% base-accuracy
regime** where selective prediction has error to remove. Baselines (all conformalized
identically, so the comparison is *which nonconformity score*): raw terminal energy `Ē*`/`E_min`,
energy spread, MSP + temperature scaling, MC-dropout, deep ensembles. Metrics: AURC (primary),
selective accuracy @ fixed coverage, ECE, guarantee-validity plot, halting compute-vs-accuracy
Pareto; paired bootstrap on ΔAURC, ≥5 seeds. Details + status in `EXPERIMENTS.md`.

## Positioning

See `RELATED_WORK.md`. One line: EBT owns scalar-energy confidence + adaptive compute but has
no restarts, no landscape geometry, and no calibration guarantee; curvature↔calibration work is
over weights, not the per-input latent energy; no prior work turns inference-time landscape
geometry into a distribution-free selective certificate for an EBM reasoner.
