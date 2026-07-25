"""Run a single named experiment from a config and append a ledger row. [Phase 3]

    PYTHONPATH=src python experiments/run_experiment.py configs/experiments/<name>.toml

Trains the reasoner, runs the disjoint-fold geometry-vs-energy selective-prediction evaluation
(``edc.eval.evaluate``), and appends one ``split="selective"`` row to results/ledger.jsonl. The
falsification verdict (does geometry beat raw energy in ΔAURC?) and the LTT abstention guarantee
live in that row's ``metrics``; figures/tables regenerate from it (invariant 6).

``run_and_append`` is the shared train->evaluate->ledger step; ``experiments/run_sweep.py`` reuses
it for every grid cell.
"""

from __future__ import annotations

import datetime as _dt
import sys
import uuid

from edc.config import load_config
from edc.eval.evaluate import evaluate
from edc.ledger import RunRecord, append
from edc.registry import build_task


def run_and_append(cfg, task, *, include_ood: bool = True, quiet: bool = False) -> dict:
    """Train + evaluate ``cfg`` and append one ``split="selective"`` ledger row; return the row.

    The single source of truth for turning a resolved config into a recorded result — used by both
    the single-experiment CLI and the sweep runner so they cannot drift. Stamps wall-clock time
    here (keeping ``edc.ledger`` pure).
    """
    m = evaluate(cfg, task, include_ood=include_ood)
    if not quiet:
        _print_summary(cfg, m)
    row = append(RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        resolved_config=cfg.to_dict(),
        task=task.name,
        split="selective",
        seed=cfg.run.seed,
        metrics=m,
    ))
    return row


def _print_summary(cfg, m: dict) -> None:
    d, lo, hi = m["delta_aurc_vs_best_energy"]
    verdict = "GEOMETRY WINS" if m["geometry_wins"] else "not separated (CI includes 0)"
    ood = f"  OOD acc={m['accuracy_ood']:.3f}" if "accuracy_ood" in m else ""
    print(f"[edc] K={m['k_restarts']} seed={cfg.run.seed}  ID acc={m['accuracy_id']:.3f}{ood}  "
          f"base error={m['base_error']:.3f}")
    print(f"[edc] AURC geometry={m['aurc']['geometry']:.4f}  "
          f"best energy ({m['best_energy_baseline']})={m['aurc'][m['best_energy_baseline']]:.4f}")
    print(f"[edc] ΔAURC(best energy − geometry)={d:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")
    print(f"[edc] falsification: {verdict}")
    ltt_b = m["ltt"]
    print(f"[edc] LTT @ alpha={ltt_b['alpha']}, delta={ltt_b['delta']}: "
          f"coverage={ltt_b['coverage']:.3f}  selective_risk={ltt_b['selective_risk']:.3f}  "
          f"within_budget={ltt_b['risk_within_budget']}")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_experiment.py <config.toml>", file=sys.stderr)
        return 2
    cfg = load_config(argv[0])
    task_kwargs = cfg.to_dict().get("task", {}).get(cfg.run.task, {})
    task = build_task(cfg.run.task, **task_kwargs)

    print(f"[edc] experiment '{cfg.run.task}' (seed={cfg.run.seed}) — training + evaluating ...")
    row = run_and_append(cfg, task)
    print(f"[edc] appended ledger row run_id={row['run_id']} -> results/ledger.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
