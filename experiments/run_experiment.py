"""Run a single named experiment from a config and append a ledger row. [Phase 3]

    PYTHONPATH=src python experiments/run_experiment.py configs/experiments/<name>.toml

Trains the reasoner, runs the disjoint-fold geometry-vs-energy selective-prediction evaluation
(``edc.eval.evaluate``), and appends one ``split="selective"`` row to results/ledger.jsonl. The
falsification verdict (does geometry beat raw energy in ΔAURC?) and the LTT abstention guarantee
live in that row's ``metrics``; figures/tables regenerate from it (invariant 6).
"""

from __future__ import annotations

import datetime as _dt
import sys
import uuid

from edc.config import load_config
from edc.eval.evaluate import evaluate
from edc.ledger import RunRecord, append
from edc.registry import build_task


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_experiment.py <config.toml>", file=sys.stderr)
        return 2
    cfg = load_config(argv[0])
    task_kwargs = cfg.to_dict().get("task", {}).get(cfg.run.task, {})
    task = build_task(cfg.run.task, **task_kwargs)

    print(f"[edc] experiment '{cfg.run.task}' (seed={cfg.run.seed}) — training + evaluating ...")
    m = evaluate(cfg, task)

    d, lo, hi = m["delta_aurc_vs_best_energy"]
    verdict = "GEOMETRY WINS" if m["geometry_wins"] else "not separated (CI includes 0)"
    print(f"[edc] ID acc={m['accuracy_id']:.3f}  OOD acc={m['accuracy_ood']:.3f}  "
          f"base error={m['base_error']:.3f}")
    print(f"[edc] AURC geometry={m['aurc']['geometry']:.4f}  "
          f"best energy ({m['best_energy_baseline']})={m['aurc'][m['best_energy_baseline']]:.4f}")
    print(f"[edc] ΔAURC(best energy − geometry)={d:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
    print(f"[edc] falsification verdict: {verdict}")
    ltt_b = m["ltt"]
    cov = ltt_b["coverage"]
    print(f"[edc] LTT @ alpha={ltt_b['alpha']}, delta={ltt_b['delta']}: "
          f"coverage={cov:.3f}  selective_risk={ltt_b['selective_risk']:.3f}  "
          f"within_budget={ltt_b['risk_within_budget']}")

    row = append(RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        resolved_config=cfg.to_dict(),
        task=task.name,
        split="selective",
        seed=cfg.run.seed,
        metrics=m,
    ))
    print(f"[edc] appended ledger row run_id={row['run_id']} -> results/ledger.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
