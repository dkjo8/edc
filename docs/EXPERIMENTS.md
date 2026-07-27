# Experiments Catalog

Every experiment: config file → purpose → expected figure/table → status. Runs append to
`results/ledger.jsonl`; figures/tables regenerate from it (`make figures`, `make tables`).

| ID | Config | Purpose | Output | Status |
|----|--------|---------|--------|--------|
| E0 | `configs/smoke.toml` | Liveness: train→K-restart infer→decode on CPU <60s | ledger row | ✅ Phase 1 |
| D1 | `configs/smoke.toml` (`edc.cli geometry`) | Per-feature correct-vs-incorrect AUROC, geometry vs raw energy — F5 precursor | ledger row | ✅ Phase 2 |
| E1 | `configs/experiments/arithmetic_selective.toml` | Geometry vs baselines, selective prediction on arithmetic | F2, F3, T1 | ✅ Phase 3 |
| E2 | `configs/experiments/graph_selective.toml` (+`sweeps/graph_seeds.toml`) | Generalization: geometry vs energy on graph shortest-path | T1 | ✅ Phase 4d |
| E3 | (logic) | Same on logic | F2, T1 | ⏳ Phase 4 |
| E4 | (hard sudoku) | Stronger-result task, OOD split | F2, F6, T1 | ⏳ Phase 4 |
| S1 | `configs/sweeps/k_restarts.toml` | K=1..16: does restart geometry add signal? | S1 fig, T2 | ✅ Phase 4a |
| H1 | `configs/experiments/arithmetic_halting.toml` | Adaptive halting: CRC compute-vs-accuracy | F4 | ✅ Phase 4b |
| A1 | (ablation grid) | Feature-group leave-one-out; τ, curvature fidelity | T2 | ⏳ Phase 4 |

## Figures → claims (mirrored in `paper/README.md`)

- **F1** method schematic (encode → K-restart descent → geometry → certificate) — TikZ, not data.
- **F2** risk–coverage curves per nonconformity score — **core selective-prediction result**. ✅
- **F3** empirical coverage vs nominal 1−α — calibration validity (must sit on the diagonal). ✅
- **F4** compute–accuracy tradeoff under adaptive halting. ✅
- **F5** geometry-feature diagnostics: basin/curvature distributions, correct vs incorrect. ✅
- **F6** distribution-shift stress: selective risk holds ID, breaks under OOD → motivates abstention. ✅
- **T1** main results (accuracy, coverage, abstention rate, avg compute).
- **T2** ablations (feature leave-one-out, K sweep).
- **T3** reproducibility appendix (seeds, configs, env) — auto-generated from ledger.

## Falsification test (wired into E1–E4)

Paired-bootstrap 95% CI on `ΔAURC(raw terminal energy − geometry)`. If it includes 0, the core
claim fails. Keep the raw-energy baseline in every selective experiment (invariant 8).

**E1 result (2026-07-24, run_id `c1557364f51d`, seed 0):** ID base acc 0.81 (regime ✓). The learned
geometry mapper **beats the best raw-energy baseline**: AURC 0.071 vs 0.145, `ΔAURC = +0.074`, 95%
CI `[+0.054, +0.093]` — **excludes 0, geometry wins.** LTT abstention at α=0.1, δ=0.05 answers 69%
of inputs at selective risk 0.075 ≤ 0.1 (base error 0.19 → the guarantee genuinely abstains). This
*reverses* the Phase-2 single-feature picture where `energy/std` out-separated any lone geometry
feature — the signal is in the **combined** geometry vector, exactly the paper's claim. Regenerate
F2/F3 with `make figures`. Single-seed so far; the ≥5-seed paired bootstrap is Phase 4.

**S1 K-ablation (2026-07-25, 5 seeds/K, reduced folds n_test=600):** the geometry lift over the best
raw-energy baseline **grows monotonically with restarts** — exactly the mechanism the method claims:

| K | AURC(geom) | ΔAURC (energy−geom) | seeds with CI excl. 0 |
|---|-----------|---------------------|-----------------------|
| 1 (≈EBT) | 0.123 | +0.026 ± 0.016 | **1/5** |
| 2 | 0.104 | +0.043 ± 0.024 | 3/5 |
| 4 | 0.082 | +0.062 ± 0.035 | 4/5 |
| 8 | 0.076 | +0.070 ± 0.033 | **5/5** |
| 16 | 0.051 | +0.083 ± 0.035 | **5/5** |

K=1 — a single descent with no basin geometry (≈ the EBT baseline) — barely separates (1/5 seeds);
the **restart** geometry is what carries the signal. This confirms the secondary falsifier does *not*
fire: K>1 gives clear lift over K=1. (T2 also lists a K=12 row from the two full-fold E1 runs, which
sits on the same trend.) Regenerate T1/T2/T3 with `make tables`, the figure with `make figures`.

## Adaptive halting — the second guarantee (H1 → F4)

`L(τ) = 1[early-stopped best-of-N answer ≠ full-budget best-of-N answer]`, stopping when basin
agreement crosses τ; CRC picks the most compute-saving τ with `E[L] ≤ α`.

**H1 result (2026-07-25, run_id `685d46f13e58`, seed 0):** CRC chose τ̂=0.917 at α=0.1 and the
empirical risk was **monotone in τ** (CRC's assumption held). At that operating point the descent
uses **42% of the step budget (57.8% compute saved)** with a halting risk of **0.9% ≤ α=10%** and
**no accuracy loss** (full 0.804 → halted 0.807). F4 is the compute-vs-accuracy Pareto; the CRC
point sits at the elbow. Both guarantees are now live: LTT abstention (E1) and CRC halting (H1).

## OOD stress — the guarantee breaks under shift (F6, run_id `1acd65c19072`)

The LTT selective-risk threshold is calibrated on the **ID** calib fold and applied to the **OOD**
test fold (magnitude shift `ood_max_operand`, OOD acc 0.21). At α=0.1 the achieved selective risk is
**0.075 ID (valid) vs 0.762 OOD** — the ID-calibrated threshold answers 66% of OOD inputs at 76%
error, so the guarantee is void once exchangeability breaks. F6 plots achieved-vs-target risk: ID on
the diagonal, OOD far above. This is the concrete argument for abstention/shift-detection in
critical systems (the guarantees are *marginal under exchangeability*, by design). **F5** shows the
mechanism: basin-agreement features (entropy, ρ) separate correct from incorrect best on arithmetic;
curvature separates weakly here — reported honestly.

## E2 — generalization to a second reasoning family (graph shortest-path)

Does the discovery replicate beyond additive arithmetic? Graph shortest-path length (BFS label,
larger-graph OOD) is a relational/combinatorial family; tuned to ID ≈ 0.70 (majority baseline ≈ 0.41).
**Result (2026-07-26, 5 seeds, `sweeps/graph_seeds.toml`):** geometry beats the best raw-energy
baseline on **5/5 seeds** — AURC 0.159 vs 0.270, `ΔAURC = +0.111 ± 0.054`, every seed's 95% CI
excludes 0. Alongside arithmetic (ΔAURC +0.083 ± 0.034, 5/5), the T1 table now shows the discovery
holds across **two structurally distinct tasks** — it is not an arithmetic artifact. (Caveat: on
graph the LTT abstention at α=0.1 with reduced calib folds certifies zero coverage — conservative,
not a failure; ΔAURC, the threshold-free ranking metric, is the generalization evidence.)

## Baseline stress test — geometry vs standard softmax confidence (Phase 4e) ⚠️ **key caveat**

Until now geometry was only compared against scalar **energy**. Adding the standard
softmax-confidence baselines the plan calls for — **MSP, temperature-scaled MSP, predictive
entropy** (from the decoder logits) — sharply bounds the claim. Over 5 seeds per task
(`arith_seeds.toml` / `graph_seeds.toml`):

| task | ΔAURC vs best **energy** (5 seeds) | ΔAURC vs best **overall baseline** (5 seeds) |
|------|-----------------------------------|----------------------------------------------|
| arithmetic | **+0.085 ± 0.036** — geometry wins 5/5 | **−0.011 ± 0.006** — geometry wins **0/5** |
| graph | **+0.111 ± 0.054** — geometry wins 5/5 | **−0.034 ± 0.008** — geometry wins **0/5** |

**Verdict.** The invariant-8 sacred test holds: geometry beats the EBT **scalar energy** on both
tasks, all seeds. But the *broader* claim does **not** survive — a plain softmax-confidence baseline
(MSP / entropy / temperature) **ties geometry on arithmetic and beats it on graph** (F2 shows
geometry, MSP, and entropy overlapping at the bottom, all far below energy). So: **landscape geometry
beats scalar energy but is not a better nonconformity score than softmax confidence** under the
current reasoner. This is a real, reported limitation, not a bug.

**Which geometry features drive it? (A1 feature ablation → T2b, 5 seeds/task).** The geometry
vector *includes* the 3 energy statistics, so does the win over raw energy just re-use energy?
Leave-one-group-out AURC (lower = better) says no:

| subset | arithmetic AURC | graph AURC |
|--------|-----------------|-----------|
| full (14 feats) | 0.059 | 0.159 |
| **drop energy** | **0.062** | **0.162** |
| drop curvature | 0.063 | 0.160 |
| drop dynamics | 0.064 | 0.187 |
| drop basin | **0.110** | **0.175** |
| *(best raw energy)* | *0.144* | *0.270* |

`drop_energy ≈ full` and both far below raw energy → the win is **genuinely from the non-energy
features**, not re-using energy. **Basin agreement is the dominant driver** on both tasks (removing
it nearly doubles arithmetic AURC toward the energy level); descent **dynamics** helps too (most on
graph); curvature adds little here. This is the mechanism behind the invariant-8 win — and it is
consistent with F5. (It does not change the 4e caveat: basin/dynamics beat *energy* but not softmax.)

### Phase 4g — IRED learned landscape: it works, and it flips the softmax caveat ✅

The lever from Phase 4e: replace the supervised basin-center bowl with a genuinely learned,
input-conditioned multi-basin energy (opt-in `[train] objective="ired"`: a learned per-class latent
**codebook** as attractors, `mlp_ebm energy_form="learned"`), and see whether restart geometry then
beats softmax confidence, not just scalar energy.

**Getting it to reason.** Two failed attempts first: denoising score matching toward the anchors was
**stuck** (`dsm ≈ 16` of a trivial ≈ 24, flat across epochs and samplers), and an added **annealed
Langevin** sampler (`[inference] sampler="annealed"`, a geometric explore→settle schedule) did not
help on its own. The fix was the objective: an **IREM-style contrastive + stationarity** loss
(`train/losses.ired_loss_fn`). Contrastive pushes `E(μ_y)` below the wrong-class anchor and below a
random `z0` by a margin (reachability); the **stationarity** term drives `‖∇_z E(μ_y)‖→0`
(weight `ired_stat_weight`) so anchors are genuine \emph{local minima}—the piece the DSM attempts
lacked (low-energy but not stationary, so descent never settled). With annealed descent, ID accuracy
rises from chance to $0.72$–$0.90$.

**Result (5 seeds, learned IRED landscape; `sweeps/ired_seeds.toml`).**

| reasoner | ΔAURC vs scalar energy | ΔAURC vs best **softmax** baseline |
|----------|------------------------|------------------------------------|
| basin-center (Phase-1) | $+0.085$, geometry wins $5/5$ | $-0.011$, geometry wins **$0/5$** |
| **IRED (learned)** | $+0.159\pm0.051$, wins $5/5$ | $\mathbf{+0.007\pm0.005}$, wins **$4/5$** |

Under a genuinely learned multi-basin landscape, **geometry beats the best softmax-confidence
baseline (MSP / temperature / entropy) on $4/5$ seeds** (only seed~3 ties), where the basin-center
reasoner lost $0/5$. The margin is small but consistent, and it **flips the Phase-4e caveat**: the
limiter was the simple landscape, not the geometry idea---exactly the hypothesis. LTT abstention
remains valid under IRED (e.g. seed~0: $81\%$ coverage at selective risk $0.052\le\alpha$). Default
`basin_center` is unchanged; this is opt-in and additive. Next: multi-seed IRED on graph, and a
larger stat-weight / capacity sweep to widen the margin.

**Likely cause of the softmax tie + the lever.** The reasoner is the Phase-1 **supervised
basin-center** model with a
shallow decoder — its softmax is already informative, leaving little for geometry to add beyond
energy. The research plan's **Phase-4 IRED annealed-landscape / score-matching training** is the
intended fix: a genuinely learned, multi-basin landscape should make restart geometry carry signal
the softmax does not. Re-running this baseline stress test under IRED training is the priority next
experiment (see SESSION_HANDOFF). MC-dropout / deep ensembles remain deferred (the K-restart
best-of-N already supplies ensemble-like diversity).

## Operating regime

Tune each task's difficulty so base accuracy is **70–85%**. Fully-solved tasks saturate AURC and
make every method tie — a null result by construction, not by science.
