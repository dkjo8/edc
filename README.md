# Energy Descent Certificates (EDC)

> **Critical thinking, for critical systems.** — [Polished Snow](https://plsn.ai)

An energy-based reasoner answers by **optimizing** a learned energy `E(h_x, z)` over a
latent — it *descends* to an answer. This project asks a question nobody has answered:

> **Is the _geometry_ of that descent a calibrated signal of whether the answer is right —
> and can we turn it into a distribution-free guarantee?**

We run the inference-time optimization from **K stochastic restarts** and read off the
*shape* of the landscape it lands in:

- **basin agreement** — do independent restarts converge to the same answer?
- **curvature** — is the optimum a sharp spike or a wide, flat basin? (Hessian eigenvalues
  of the energy **over the latent**, per input, via Hessian-vector products)
- **descent dynamics** — how fast, how monotonically, how far did it settle?

Wrapped in distribution-free risk control, these features give a **certificate**:

- **Selective prediction / abstention** — answer only when a *guaranteed* error rate holds
  among answered inputs; otherwise **abstain** and route to a human/verifier
  (Learn-then-Test).
- **Adaptive halting** — stop "thinking" as soon as a *guaranteed* error budget is met
  (Conformal Risk Control).

The scientific claim — and the thing this repo is built to falsify — is that **landscape
geometry beats the scalar energy** (the signal Energy-Based Transformers already use) as a
predictor of correctness. See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

## Why this is new

Energy-Based Transformers (arXiv `2507.02092`) already use *scalar energy* as confidence and
*think longer* on hard inputs — but with **no restarts, no curvature/basin geometry, and no
calibration guarantee**. Curvature↔calibration work studies curvature of the *loss over
weights*; ours is curvature of the *energy over the latent, per input*. No prior work turns
inference-time landscape geometry into a distribution-free selective certificate for an EBM
reasoner. Full positioning in [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Quickstart

```bash
uv sync --python 3.12 --extra dev     # pinned CPU env, no GPU needed

make test        # offline, CPU-only test suite
make smoke       # end-to-end base reasoner on CPU (<60s); appends a ledger row
```

Run the reasoner directly:

```bash
PYTHONPATH=src python -m edc.cli smoke --config configs/smoke.toml
```

## Layout

```
src/edc/
  energy/       encoder → E_θ(h_x, z) → decoder            (JAX/Flax)
  inference/    K-restart Langevin descent + trajectory logging
  geometry/     basin · curvature (HVP) · energy stats · dynamics   ✓
  conformal/    Learn-then-Test (abstention) · Conformal Risk Control (halting)  [Phase 3]
  halting/      adaptive compute policy                    [Phase 3]
  tasks/        synthetic reasoning families (+ OOD splits)
  train/        IREM/IRED-style energy training
  eval/         AURC · risk–coverage · ECE · compute-vs-accuracy     [Phase 3]
configs/  experiments/  analysis/  results/ledger.jsonl  tests/  docs/  paper/
```

## Status

Phase 0 (skeleton), Phase 1 (runnable base reasoner), and Phase 2 (geometry features) are in.
`edc.cli geometry` scores each feature's correct-vs-incorrect AUROC against the raw-energy
baseline. The conformal layer, full experiments, and the paper are scaffolded and tracked in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Current state:
[`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md).

## License

Unreleased research code. All rights reserved until a license is added.
