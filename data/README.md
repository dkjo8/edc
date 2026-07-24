# data/

All task data is **synthetic and regenerated from a seed + config** — nothing here is tracked
except this README.

- `raw/` — generated problem instances (gitignored). Produced on demand by the task generators
  in `src/edc/tasks/` via `edc.seeding.numpy_rng(seed, ...)`.
- `processed/` — cached calibration/eval splits (gitignored).

To reproduce any dataset: re-run the experiment with the same `run.seed` and the same task
config. The generators are pure functions of `(seed, split, n)`, so the data is exact.
