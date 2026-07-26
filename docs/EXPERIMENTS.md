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

## Operating regime

Tune each task's difficulty so base accuracy is **70–85%**. Fully-solved tasks saturate AURC and
make every method tie — a null result by construction, not by science.
