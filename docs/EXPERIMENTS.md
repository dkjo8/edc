# Experiments Catalog

Every experiment: config file → purpose → expected figure/table → status. Runs append to
`results/ledger.jsonl`; figures/tables regenerate from it (`make figures`, `make tables`).

| ID | Config | Purpose | Output | Status |
|----|--------|---------|--------|--------|
| E0 | `configs/smoke.toml` | Liveness: train→K-restart infer→decode on CPU <60s | ledger row | ✅ Phase 1 |
| D1 | `configs/smoke.toml` (`edc.cli geometry`) | Per-feature correct-vs-incorrect AUROC, geometry vs raw energy — F5 precursor | ledger row | ✅ Phase 2 |
| E1 | `configs/experiments/arithmetic_selective.toml` | Geometry vs baselines, selective prediction on arithmetic | F2, F3, T1 | ✅ Phase 3 |
| E2 | (graph) | Same on graph planning | F2, T1 | ⏳ Phase 4 |
| E3 | (logic) | Same on logic | F2, T1 | ⏳ Phase 4 |
| E4 | (hard sudoku) | Stronger-result task, OOD split | F2, F6, T1 | ⏳ Phase 4 |
| S1 | `configs/sweeps/k_restarts.toml` | K=1..32: does restart geometry add signal? | F5, T2 | ⏳ Phase 4 |
| A1 | (ablation grid) | Feature-group leave-one-out; τ, curvature fidelity | T2 | ⏳ Phase 4 |

## Figures → claims (mirrored in `paper/README.md`)

- **F1** method schematic (encode → K-restart descent → geometry → certificate) — TikZ, not data.
- **F2** risk–coverage curves per nonconformity score — **core selective-prediction result**.
- **F3** empirical coverage vs nominal 1−α — calibration validity (must sit on the diagonal).
- **F4** compute–accuracy tradeoff under adaptive halting.
- **F5** geometry-feature diagnostics: basin/curvature distributions, correct vs incorrect.
- **F6** distribution-shift stress: coverage/abstention under OOD difficulty.
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

## Operating regime

Tune each task's difficulty so base accuracy is **70–85%**. Fully-solved tasks saturate AURC and
make every method tie — a null result by construction, not by science.
