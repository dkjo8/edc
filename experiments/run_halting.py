"""Run the adaptive-halting experiment and append a ledger row. [Phase 4b]

    PYTHONPATH=src python experiments/run_halting.py configs/experiments/arithmetic_halting.toml

Trains the reasoner, CRC-calibrates the basin-agreement halting threshold, and appends one
``split="halting"`` row with the compute-vs-accuracy metrics + tau-sweep. F4 regenerates from it
(invariant 6).
"""

from __future__ import annotations

import datetime as _dt
import sys
import uuid

from edc.config import load_config
from edc.eval.evaluate_halting import evaluate_halting
from edc.ledger import RunRecord, append
from edc.registry import build_task


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_halting.py <config.toml>", file=sys.stderr)
        return 2
    cfg = load_config(argv[0])
    task_kwargs = cfg.to_dict().get("task", {}).get(cfg.run.task, {})
    task = build_task(cfg.run.task, **task_kwargs)

    print(f"[edc] halting '{cfg.run.task}' (seed={cfg.run.seed}) — training + calibrating ...")
    m = evaluate_halting(cfg, task)

    tau = m["tau_hat"]
    tau_s = f"{tau:.3f}" if tau is not None else "None (budget infeasible)"
    print(f"[edc] CRC tau_hat={tau_s}  @ alpha={m['alpha']}  (risk monotone in tau: "
          f"{m['risk_monotone_in_tau']})")
    print(f"[edc] compute used={m['compute_used']:.3f} (saved {m['compute_saved']:.3f})  "
          f"halting risk={m['halting_risk']:.3f}  within_budget={m['risk_within_budget']}")
    print(f"[edc] accuracy: full={m['full_accuracy']:.3f}  halted={m['halted_accuracy']:.3f}  "
          f"drop={m['accuracy_drop']:+.3f}")

    row = append(RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        resolved_config=cfg.to_dict(),
        task=task.name,
        split="halting",
        seed=cfg.run.seed,
        metrics=m,
    ))
    print(f"[edc] appended ledger row run_id={row['run_id']} -> results/ledger.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
