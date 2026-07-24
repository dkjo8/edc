# Benchmarks

Headline numbers, read from `results/ledger.jsonl`. Populated as experiments land; do not edit
by hand — regenerate from the ledger.

## Phase 1 — base reasoner liveness (smoke)

| Task | Split | Metric | Value |
|------|-------|--------|-------|
| arithmetic | id | best-of-N acc | _run `make smoke`_ |
| arithmetic | id | majority acc | _run `make smoke`_ |
| arithmetic | ood | best-of-N acc | _run `make smoke`_ |

Chance = 1/modulus = 0.10. The smoke config is tiny (fast, not tuned) — it exists to prove the
pipeline runs end-to-end and beats chance, not to report a headline result.

## Phase 3/4 — the numbers that matter (pending)

- ΔAURC(raw energy − geometry) with 95% CI per task (the falsification test).
- Selective accuracy @ 80%/90% coverage; empirical vs nominal coverage.
- Compute saved at matched error under adaptive halting.
