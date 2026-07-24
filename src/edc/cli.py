"""EDC command-line entry point.

Subcommands:
  smoke   Train a tiny reasoner and run K-restart inference end-to-end on CPU (<60s),
          then append a row to results/ledger.jsonl. This is the Phase-1 liveness check.

Later phases add `run` (full experiment), `sweep`, and `figures` subcommands; those live in
experiments/ and analysis/ and are wired here as they land.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import uuid

import jax.numpy as jnp
import numpy as np

from edc.config import load_config
from edc.inference import restarts
from edc.ledger import RunRecord, append
from edc.registry import build_task
from edc.seeding import numpy_rng, root_key
from edc.train.train_ebm import train


def _accuracy(fns, params, cfg, task, split, key):
    rng = numpy_rng(cfg.run.seed, 7, 0 if split == "id" else 1)
    batch = task.sample(rng, cfg.eval.n_eval, split=split)
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, key)
    pred_boN, _ = restarts.best_of_n_energy(traj)
    pred_maj = restarts.majority_vote(traj)
    y = batch.y
    return {
        "best_of_n_acc": float(np.mean(task.evaluate(np.asarray(pred_boN), y))),
        "majority_acc": float(np.mean(task.evaluate(np.asarray(pred_maj), y))),
        "mean_terminal_energy": float(jnp.mean(traj.terminal_energy)),
    }


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    task = build_task(cfg.run.task, **cfg.to_dict().get("task", {}).get(cfg.run.task, {}))

    print(f"[edc] training on '{task.name}' (seed={cfg.run.seed}) ...")
    params, fns, history = train(cfg, task)
    last = history["epochs"][-1]
    print(f"[edc] final train loss={last['loss']:.4f} ce={last['ce']:.4f} "
          f"center_norm={last['center_norm']:.3f}")

    key = root_key(cfg.run.seed)
    id_metrics = _accuracy(fns, params, cfg, task, "id", key)
    ood_metrics = _accuracy(fns, params, cfg, task, "ood", key)

    chance = 1.0 / task.n_classes
    print(f"[edc] ID  best-of-N acc={id_metrics['best_of_n_acc']:.3f} "
          f"majority acc={id_metrics['majority_acc']:.3f}  (chance={chance:.3f})")
    print(f"[edc] OOD best-of-N acc={ood_metrics['best_of_n_acc']:.3f}")

    metrics = {
        "chance": chance,
        "id": id_metrics,
        "ood": ood_metrics,
        "k_restarts": cfg.inference.k_restarts,
        "steps": cfg.inference.steps,
        "final_train_loss": last["loss"],
    }
    row = append(RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        resolved_config=cfg.to_dict(),
        task=task.name,
        split="id+ood",
        seed=cfg.run.seed,
        metrics=metrics,
    ))
    print(f"[edc] appended ledger row run_id={row['run_id']} -> results/ledger.jsonl")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edc", description="Energy Descent Certificates")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("smoke", help="end-to-end base-reasoner smoke (CPU)")
    ps.add_argument("--config", default="configs/smoke.toml")
    ps.set_defaults(func=cmd_smoke)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
