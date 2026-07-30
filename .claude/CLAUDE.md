# EDC — operating rules & hard invariants

Energy Descent Certificates (EDC): turning the **geometry of an energy-based reasoner's
inference-time descent** into distribution-free reliability certificates. Read
`docs/RESEARCH_PLAN.md` for the science, `docs/SESSION_HANDOFF.md` for current state.

## Hard invariants (do not violate)

1. **Framework:** JAX is the numeric core. It may only appear under `src/edc/energy/*`,
   `src/edc/inference/*`, and `src/edc/geometry/curvature.py`. Everything else is plain
   Python so the core stays swappable behind `edc.energy.base`.
2. **Env:** `uv sync --python 3.12` only. Run code as `PYTHONPATH=src python ...`.
3. **Tests are offline + CPU-only.** No GPU, no network, no model downloads in `tests/`.
   `make test` must stay green. Set `JAX_PLATFORMS=cpu` in test/CI paths.
4. **Determinism:** all randomness flows through `edc.seeding`. Same seed → identical
   trajectory. Never call `jax.random.PRNGKey` ad hoc outside `seeding.py` helpers.
5. **Ledger is append-only.** Never hand-edit `results/ledger.jsonl`; append via
   `edc.ledger`. Every run records its fully-resolved config + git sha + env.
6. **Figures are regenerable.** Every figure/table comes from `analysis/make_figures.py`
   / `make_tables.py` reading the ledger — never from a one-off script or manual step.
7. **Conformal validity:** the nonconformity mapper is fit on a fold **disjoint** from the
   calibration fold. Never calibrate on training data. Guarantees hold only under
   exchangeability — say so wherever a guarantee is claimed.
8. **The falsification test is sacred:** geometry must beat raw terminal energy in ΔAURC.
   Keep the raw-energy baseline wired into every selective-prediction experiment.

## Phase map (see docs/EXPERIMENTS.md)

- Phase 0 skeleton ✅ / Phase 1 base reasoner ✅ / Phase 2 geometry features ✅
- Phase 3 conformal falsification + LTT abstention ✅ (E1: geometry beats energy, ΔAURC CI
  excludes 0; F2/F3 regenerate from the ledger).
- Phase 4a robustness + K-ablation ✅ (`run_sweep` + `aggregate` + `make_tables` T1/T2/T3 + S1
  K-lift figure; S1: geometry lift grows monotonically with restarts, K=1≈EBT barely separates).
- Phase 4b adaptive halting ✅ (opt-in per-step decode + `halting.adaptive` CRC + F4; H1: ~58%
  compute saved at halting risk ≤ α, no accuracy loss). Both guarantees now live.
- Phase 4c arithmetic figure set complete ✅ (F5 mechanism + F6 OOD stress; F6: selective risk
  0.075 ID vs 0.762 OOD — guarantee breaks under shift, motivating abstention).
- Phase 4d generalization ✅ (graph shortest-path task; E2: geometry beats energy 5/5 seeds,
  ΔAURC +0.111 — discovery replicates on a 2nd structurally distinct family; per-task T1).
- Phase 4e baseline stress test ✅ ⚠️ (MSP/temp/entropy; geometry beats scalar energy 5/5 but
  does NOT beat softmax confidence — ties arith, loses graph). Key caveat; see EXPERIMENTS.md.
- Phase 4f feature-group ablation ✅ (A1/T2b; basin agreement is the dominant driver, drop_energy≈full
  so the win is non-energy features — rebuts "it's just energy").
- Phase 5 paper draft ✅ (`paper/sections/*.tex` written from the real results; figures/tables wired;
  compiles via `tectonic`).
- Phase 4g/4h/4i IRED landscape ✅ (opt-in `objective="ired"` contrastive+stationarity + annealed
  sampler; it reasons). Under the learned landscape geometry beats softmax **4/5 on arithmetic**
  (flips the 4e caveat) but **ties on graph — even with a strong ID~0.83 reasoner (0/5)**, so the
  flip is **genuinely task-dependent, not a reasoner-quality artifact** (I tested that). Aggregation
  is objective-aware so IRED never pollutes the basin-center T1.
- Phase 4j complementarity ✅ (T4): geometry adds *conditional* signal over softmax ONLY where it
  wins head-to-head (arithmetic-IRED 4/5); on graph-IRED the combined [geom+softmax] mapper ≈ softmax
  (0/5) — the graph tie is **redundancy, not complementarity**.
- Phase 4k richer geometry ✅ (T5): opt-in `[eval] richer_geometry=true` appends the full Hessian
  spectrum (`spectrum/*`, exact d×d eigh) + mode-connectivity barriers (`connect/*`) after the base
  14. Re-ran the 4j test on all 4 cells: richer geometry helps **nowhere** — graph-IRED conditional
  signal stays −0.002 (0/5). So the **graph redundancy is a task property, not thin features**; it's
  a wash where geometry already wins (arith-IRED +0.007 4/5) and slightly worse on fixed-bowl cells.
  Aggregation is feature-set-aware (`feature_set_of`, canonical tables default `feature_set="base"`)
  so richer never pollutes T1/T2/T4. JAX for spectrum/connectivity lives in `curvature.py` (inv. 1).
- Phase 4l certificates on firm footing ✅ (T6/T7): put the guarantees on a 5-seed footing.
  `run_halting_sweep.py` + `aggregate.{halting_rows,aggregate_halting_cell}` → **T6**: arith halting
  saves 57.1%±1.7% (was single-seed 57.8%), graph 80.8%±5.2%, both risk ≤ α, within-budget 5/5.
  `[sweep] include_ood=true` + OOD aggregation block → **T7**: guarantee breaks under shift over 5
  seeds (arith ID risk 0.077→OOD 0.764, 0/5; graph 0.010→0.168, 4/5). Full-fold graph cert run: LTT
  certifies 1.1% cov at α=0.1 but **22.5% at α=0.15 / 48.5% at α=0.20, validity holds at every α** —
  the "certifies zero" was an operating-point artifact of graph's ~30% error floor, not a broken
  certificate. Aggregation dedups halting rows by seed (drops the stale Phase-4b H1 config-hash row);
  `plotting._selective_row`/`_halting_row` prefer arithmetic so full-fold graph rows don't flip F2/F4/F6.
- Next: E3/E4 (a 3rd/4th task to characterize *when* geometry beats softmax) · Modal · scale-up.
