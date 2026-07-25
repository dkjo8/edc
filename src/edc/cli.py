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

import jax
import jax.numpy as jnp
import numpy as np

from edc.config import load_config
from edc.eval import metrics
from edc.geometry.features import geometry_features
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


def _geometry_report(fns, params, cfg, task, split, key) -> dict:
    """Compute geometry features on a split and score each feature's correct-vs-incorrect AUROC."""
    k_solve, k_geom = jax.random.split(key)
    rng = numpy_rng(cfg.run.seed, 8, 0 if split == "id" else 1)
    batch = task.sample(rng, cfg.eval.n_eval, split=split)
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, k_solve)

    feats, names = geometry_features(fns=fns, params=params, traj=traj, key=k_geom,
                                     grad_tol=cfg.inference.grad_tol)
    pred_boN, _ = restarts.best_of_n_energy(traj)
    correct = task.evaluate(np.asarray(pred_boN), batch.y)

    auroc = {name: metrics.single_feature_auroc(feats[:, j], correct)
             for j, name in enumerate(names)}
    return {"accuracy": float(np.mean(correct)), "feature_auroc": auroc, "feature_names": names}


def _print_geometry_table(split: str, report: dict) -> None:
    sep = lambda a: abs(a - 0.5)  # noqa: E731  distance from chance = separation strength
    ranked = sorted(report["feature_auroc"].items(), key=lambda kv: sep(kv[1]), reverse=True)
    print(f"\n[edc] {split.upper()} base acc={report['accuracy']:.3f} "
          f"(best-of-N energy) — per-feature correct-vs-incorrect AUROC:")
    for name, a in ranked:
        group = "energy" if name.startswith("energy/") else "geom  "
        print(f"       {group}  {name:<20} AUROC={a:.3f}  |sep|={sep(a):.3f}")


def cmd_geometry(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    task = build_task(cfg.run.task, **cfg.to_dict().get("task", {}).get(cfg.run.task, {}))

    print(f"[edc] training on '{task.name}' (seed={cfg.run.seed}) ...")
    params, fns, history = train(cfg, task)
    print(f"[edc] final train loss={history['epochs'][-1]['loss']:.4f}")

    key = root_key(cfg.run.seed)
    k_id, k_ood = jax.random.split(key)
    reports = {"id": _geometry_report(fns, params, cfg, task, "id", k_id),
               "ood": _geometry_report(fns, params, cfg, task, "ood", k_ood)}

    sep = lambda a: abs(a - 0.5)  # noqa: E731
    for split in ("id", "ood"):
        _print_geometry_table(split, reports[split])

    # Falsification smell test: does the best geometry feature separate better than the best
    # raw-energy feature (the EBT baseline)? Full ΔAURC bootstrap arrives in Phase 3.
    auroc = reports["id"]["feature_auroc"]
    geom = {n: a for n, a in auroc.items() if not n.startswith("energy/")}
    energy = {n: a for n, a in auroc.items() if n.startswith("energy/")}
    best_geom = max(geom, key=lambda n: sep(geom[n]))
    best_energy = max(energy, key=lambda n: sep(energy[n]))
    ahead = sep(geom[best_geom]) > sep(energy[best_energy])
    verdict = "geometry ahead" if ahead else "energy ahead"
    print(f"\n[edc] ID best geometry '{best_geom}' |sep|={sep(geom[best_geom]):.3f}  vs  "
          f"best energy '{best_energy}' |sep|={sep(energy[best_energy]):.3f}  ({verdict})")

    metrics_row = {
        "id": reports["id"],
        "ood": reports["ood"],
        "best_geometry_feature": best_geom,
        "best_geometry_sep": sep(geom[best_geom]),
        "best_energy_feature": best_energy,
        "best_energy_sep": sep(energy[best_energy]),
        "k_restarts": cfg.inference.k_restarts,
        "steps": cfg.inference.steps,
    }
    row = append(RunRecord(
        run_id=uuid.uuid4().hex[:12],
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        resolved_config=cfg.to_dict(),
        task=task.name,
        split="geometry-diagnostic",
        seed=cfg.run.seed,
        metrics=metrics_row,
    ))
    print(f"[edc] appended ledger row run_id={row['run_id']} -> results/ledger.jsonl")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edc", description="Energy Descent Certificates")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("smoke", help="end-to-end base-reasoner smoke (CPU)")
    ps.add_argument("--config", default="configs/smoke.toml")
    ps.set_defaults(func=cmd_smoke)

    pg = sub.add_parser("geometry", help="Phase-2 geometry-feature separation diagnostic (CPU)")
    pg.add_argument("--config", default="configs/smoke.toml")
    pg.set_defaults(func=cmd_geometry)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
