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

**Result (5 seeds/task; `sweeps/ired_seeds.toml`, `sweeps/graph_ired_seeds.toml`).** ΔAURC vs the
best of all baselines (energy + softmax); "wins" = seeds whose 95% CI clears 0.

| task | reasoner | ΔAURC vs scalar energy | ΔAURC vs best **softmax** baseline |
|------|----------|------------------------|------------------------------------|
| arithmetic | basin-center | $+0.085$, wins $5/5$ | $-0.011$, wins **$0/5$** |
| arithmetic | **IRED (learned)** | $+0.159\pm0.051$, $5/5$ | $\mathbf{+0.007\pm0.005}$, wins **$4/5$** |
| graph | basin-center | $+0.111$, wins $5/5$ | $-0.034$, wins **$0/5$** |
| graph | IRED (weak, ID~0.58) | $+0.064$, $5/5$ | $+0.004\pm0.014$, wins $1/5$ |
| graph | **IRED (strong, ID~0.83)** | $+0.092\pm0.029$, $5/5$ | $+0.001\pm0.003$, wins **$0/5$** (clean ties) |

**Verdict — the flip is genuinely task-dependent, not a reasoner-quality artifact.** On
**arithmetic**, a learned multi-basin landscape makes geometry beat the best softmax-confidence
baseline (MSP/temp/entropy) $4/5$ seeds, reversing the Phase-4e $0/5$ caveat. The obvious explanation
for the graph tie was a weak reasoner (the first IRED graph model capped at ID $\approx0.58$), so we
**tested it**: a larger IRED graph reasoner (hidden 256 / latent 48) reaches ID $\approx0.83$ — as
strong as arithmetic — and geometry \emph{still} only ties softmax ($0/5$, ΔAURC $+0.001\pm0.003$; it
still beats scalar energy $5/5$). So the flip is **specific to arithmetic**, not a general property of
learned landscapes nor merely a function of accuracy: on graph, softmax confidence already captures
what the descent geometry would add. Honest and reported narrowly. LTT abstention stays valid under
IRED; default `basin_center` unchanged; aggregation is objective-aware so IRED never pollutes T1.

### Complementarity — is the graph tie redundancy or complementarity? (Phase 4j → T4)

"Ties softmax head-to-head" is ambiguous: geometry could be **redundant** with softmax, or
**complementary** (carrying signal softmax lacks, just not enough to win alone). To decide, we fit a
mapper on `[geometry ∪ softmax]` and compare it to a mapper on `[softmax]` alone (both on the fit
fold); positive `ΔAURC(softmax − geom+softmax)` with the CI clearing 0 means geometry adds
conditional signal. Over 5 seeds per (task, reasoner):

| task | reasoner | ΔAURC geom \emph{vs} softmax (head-to-head) | ΔAURC geom \emph{adds} over softmax |
|------|----------|--------------------------------------------|-------------------------------------|
| arithmetic | basin-center | $-0.011$ ($0/5$) | $+0.001$ ($0/5$) |
| arithmetic | **IRED** | $+0.007$ ($4/5$) | $\mathbf{+0.007}$ (**$4/5$ add**) |
| graph | basin-center | $-0.034$ ($0/5$) | $-0.006$ ($0/5$) |
| graph | **IRED** | $+0.001$ ($0/5$, ties) | $+0.001$ (**$0/5$ add**) |

**Verdict — the graph tie is redundancy, not complementarity.** Geometry adds conditional signal
over softmax **only where it also wins head-to-head** (arithmetic-IRED, $4/5$). Everywhere else,
including the crux graph-IRED cell, `[geometry+softmax]` is no better than `[softmax]` alone
(ΔAURC $\approx +0.001$, $0/5$) — a well-calibrated softmax already captures whatever descent
geometry would contribute. So landscape geometry is a genuinely useful reliability signal that (i)
always beats the EBT scalar energy and (ii) beats/adds-to softmax **only** under a learned landscape
on arithmetic; it is redundant with softmax on graph.

**Likely cause of the softmax tie + the lever.** The reasoner is the Phase-1 **supervised
basin-center** model with a
shallow decoder — its softmax is already informative, leaving little for geometry to add beyond
energy. The research plan's **Phase-4 IRED annealed-landscape / score-matching training** is the
intended fix: a genuinely learned, multi-basin landscape should make restart geometry carry signal
the softmax does not. Re-running this baseline stress test under IRED training is the priority next
experiment (see SESSION_HANDOFF). MC-dropout / deep ensembles remain deferred (the K-restart
best-of-N already supplies ensemble-like diversity).

### Richer geometry — is the graph tie thin features or a task property? (Phase 4k → T5)

The Phase-4j redundancy verdict rested on a thin curvature representation: of the 14 features,
curvature was only `λmax` (power iteration) and `tr(H)` (Hutchinson). So "geometry is redundant with
softmax on graph" was confounded with "our geometry features are impoverished." Phase 4k adds two
richer, opt-in groups (`[eval] richer_geometry=true`) — the **full Hessian spectrum**
(`spectrum/*`: `lmin`, negative-eigenvalue fraction, effective rank / participation ratio, log-det,
each mean-over-restarts and at the best restart; exact `d×d` eigendecomposition since `d` is small)
and **mode connectivity** (`connect/*`: energy barriers along the straight-line path from the best
restart to the others) — and re-runs the Phase-4j complementarity test on all four cells. Same
reasoners, folds, and mappers; only the feature set changes.

| task | reasoner | features | AURC geom | ΔAURC vs softmax | ΔAURC geom *adds* | seeds add |
|------|----------|----------|-----------|------------------|-------------------|-----------|
| arithmetic | basin-center | base | $0.059$ | $-0.011$ | $+0.001$ | $0/5$ |
| arithmetic | basin-center | **richer** | $0.061$ | $-0.012$ | $-0.002$ | $1/5$ |
| arithmetic | IRED | base | $0.028$ | $+0.007$ | $+0.007$ | $4/5$ |
| arithmetic | IRED | **richer** | $0.028$ | $+0.008$ | $+0.007$ | $4/5$ |
| graph | basin-center | base | $0.159$ | $-0.034$ | $-0.006$ | $0/5$ |
| graph | basin-center | **richer** | $0.162$ | $-0.037$ | $-0.009$ | $0/5$ |
| **graph** | **IRED** | base | $0.056$ | $+0.001$ | $+0.001$ | $0/5$ |
| **graph** | **IRED** | **richer** | $0.055$ | $+0.002$ | $\mathbf{-0.002}$ | **$0/5$** |

**Verdict — the graph redundancy is a task property, not a feature-poverty artifact.** In the crux
cell (graph-IRED), the full spectrum + mode connectivity leave the conditional signal over softmax at
$-0.002$, $0/5$ — geometry still adds *nothing*. Richer geometry changes the picture nowhere: it is a
wash on the one cell geometry already wins (arithmetic-IRED, $+0.007$, $4/5$, unchanged) and slightly
*worse* on the fixed-bowl basin-center cells (adding features to a trivial landscape only adds noise:
AURC $0.059\to0.061$ arith, $0.159\to0.162$ graph). So the answer to "does *more* landscape signal
help on graph?" is **no** — a well-calibrated softmax genuinely captures what descent geometry would
contribute there, even with the richest geometry we can extract. The base rows reproduce the Phase-4j
numbers exactly, confirming richer geometry is purely additive and non-breaking (default stays the
canonical 14 features; aggregation is feature-set-aware so richer never pollutes T1/T2/T4).

### Certificates on firm footing — multi-seed guarantees + graph certification (Phase 4l → T6/T7)

The certificates are the contribution, but three of their numbers were single-seed or vacuous. This
phase puts all three on a 5-seed footing (new `run_halting_sweep.py`; `[sweep] include_ood=true`;
objective/feature-set-style aggregation for halting + OOD; full-fold graph cert run). Additive — no
change to the geometry features or the falsification test.

**Adaptive halting (CRC), 5 seeds — T6.** The H1 headline replicates and extends:

| task | τ̂ | compute saved | halting risk (≤ α=0.1) | seeds ≤ α |
|------|-----|---------------|------------------------|-----------|
| arithmetic | $0.917$ | $0.571 \pm 0.017$ | $0.012 \pm 0.005$ | $5/5$ |
| graph | $0.85\pm0.03$ | $\mathbf{0.808 \pm 0.052}$ | $0.065 \pm 0.032$ | $5/5$ |

The single-seed 57.8% arithmetic number is now $57.1\%\pm1.7\%$ over 5 seeds (within budget every
seed); on graph the CRC halt saves **~81%** of compute at risk $0.065 \le 0.1$ — the halting
guarantee holds on both tasks across seeds.

**OOD stress, 5 seeds — T7.** The ID-calibrated selective threshold applied to the shifted fold:

| task | acc ID | acc OOD | sel. risk ID | sel. risk OOD | seeds OOD ≤ α |
|------|--------|---------|--------------|---------------|---------------|
| arithmetic | $0.83$ | $0.21$ | $0.077$ | $\mathbf{0.764}$ | $0/5$ |
| graph | $0.70$ | $0.33$ | $0.010$ | $0.168$ | $4/5$ |

The guarantee holds ID and **breaks under shift** on both tasks (arithmetic catastrophically:
$0.077\to0.764$; graph mildly: $0.010\to0.168$) — 5-seed confirmation of F6, and the argument for
treating an abstention as a routing signal.

**Graph certification — the certificate is valid on graph, at its honest operating point.** At the
full fold ($n_\text{calib}=1500$) the graph LTT abstention still certifies almost nothing at
α=0.1 (coverage $0.011$), because graph's ~30% base error leaves no confident slice with true error
below 10%. But sweeping α (from the same `coverage_validity` block; mean over 5 seeds):

| α | certified coverage | achieved risk |
|-----|--------------------|---------------|
| $0.10$ | $0.011$ | $0.010$ |
| $0.15$ | $0.225$ | $0.078$ |
| $0.20$ | $0.485$ | $0.157$ |
| $0.30$ | $0.895$ | $0.263$ |

Achieved risk sits **at or below target at every α** — validity holds; the earlier "certifies zero
coverage" was an α=0.1 operating-point artifact, not a broken certificate. Graph certifies **22.5%
coverage at α=0.15** and **48.5% at α=0.20**, versus arithmetic's 74% at α=0.10 — the honest cost of
a harder task's higher error floor, not a failure. (F3 shows the arithmetic α-sweep on the diagonal.)

## Operating regime

Tune each task's difficulty so base accuracy is **70–85%**. Fully-solved tasks saturate AURC and
make every method tie — a null result by construction, not by science.
