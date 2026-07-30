"""Expand a halting sweep grid into many adaptive-halting runs. [Phase 4l]

    PYTHONPATH=src python experiments/run_halting_sweep.py configs/sweeps/arith_halting_seeds.toml

The multi-seed counterpart of ``run_halting.py``: same sweep-TOML format as ``run_sweep`` (a
``base`` config + optional ``[sweep.override]`` + ``[sweep.grid]``), but each cell trains, CRC-
calibrates the halting threshold, and appends one ``split="halting"`` row via ``evaluate_halting`` —
so the compute-saved / halting-risk guarantee gets a multi-seed mean±std instead of a single seed.
``analysis/aggregate.halting_rows`` + ``aggregate_halting_cell`` summarise the rows (T6).
"""

from __future__ import annotations

import datetime as _dt
import sys
import uuid
from pathlib import Path

from edc.config import load_from_dict
from edc.eval.evaluate_halting import evaluate_halting
from edc.ledger import RunRecord, append
from edc.registry import build_task

try:  # sys.path[0] is experiments/ when run as a script; fall back to package form.
    from run_sweep import cell_config_dict, load_sweep
except ImportError:  # pragma: no cover
    from experiments.run_sweep import cell_config_dict, load_sweep


def run_halting_and_append(cfg, task) -> dict:
    """Train + CRC-calibrate + evaluate halting for one cell, append a ``split='halting'`` row."""
    m = evaluate_halting(cfg, task)
    return append(RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        resolved_config=cfg.to_dict(),
        task=task.name,
        split="halting",
        seed=cfg.run.seed,
        metrics=m,
    ))


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_halting_sweep.py <sweep.toml>", file=sys.stderr)
        return 2
    base, override, cells, _opts = load_sweep(argv[0])

    print(f"[edc] halting sweep '{Path(argv[0]).name}': {len(cells)} cells "
          f"(override={override or '{}'}) -> appending to results/ledger.jsonl")
    for i, cell in enumerate(cells, 1):
        d = cell_config_dict(base, override, cell)
        cfg = load_from_dict(d)
        task_kwargs = d.get("task", {}).get(cfg.run.task, {})
        task = build_task(cfg.run.task, **task_kwargs)
        print(f"\n[edc] cell {i}/{len(cells)}: {cell}")
        row = run_halting_and_append(cfg, task)
        m = row["metrics"]
        tau = m["tau_hat"]
        tau_s = f"{tau:.3f}" if tau is not None else "None"
        print(f"[edc] -> run_id={row['run_id']}  tau_hat={tau_s}  saved={m['compute_saved']:.3f}  "
              f"risk={m['halting_risk']:.3f}  within_budget={m['risk_within_budget']}")
    from edc.config import REPO_ROOT
    ledger = REPO_ROOT / "results/ledger.jsonl"
    print(f"\n[edc] halting sweep complete: {len(cells)} rows appended -> {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
