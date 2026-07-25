"""Expand a sweep grid into many runs. [Phase 4]

    PYTHONPATH=src python experiments/run_sweep.py configs/sweeps/<grid>.toml

A sweep file names a ``base`` config, optional constant ``[sweep.override]`` keys, and a
``[sweep.grid]`` of dotted-key -> list. The grid is cartesian-expanded; every cell deep-merges its
overrides onto the base, then takes the same train->evaluate->ledger path as ``run_experiment`` (one
appended ``split="selective"`` row per cell). OOD is skipped in the sweep to save compute.

    [sweep]
    base = "configs/experiments/arithmetic_selective.toml"
    [sweep.override]                       # applied to every cell
    "eval.n_eval" = 600
    [sweep.grid]                           # cartesian-expanded
    "inference.k_restarts" = [1, 2, 4, 8, 16]
    "run.seed" = [0, 1, 2, 3, 4]
"""

from __future__ import annotations

import copy
import itertools
import sys
import tomllib
from pathlib import Path

from edc.config import REPO_ROOT, load_config
from edc.registry import build_task

try:  # sys.path[0] is experiments/ when run as a script; fall back to package form.
    from run_experiment import run_and_append
except ImportError:  # pragma: no cover
    from experiments.run_experiment import run_and_append


def _set_dotted(d: dict, dotted: str, value) -> None:
    """Set ``d["a"]["b"] = value`` from the key ``"a.b"``, creating intermediate dicts."""
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a dotted-key -> list grid into ``{dotted_key: value}`` cells."""
    if not grid:
        return [{}]
    keys = list(grid)
    combos = itertools.product(*(grid[k] for k in keys))
    return [dict(zip(keys, combo, strict=True)) for combo in combos]


def load_sweep(path: str | Path) -> tuple[dict, dict, list[dict]]:
    """Return ``(base_resolved_dict, override, cells)`` from a sweep TOML."""
    with open(path, "rb") as f:
        spec = tomllib.load(f)["sweep"]
    base = load_config(spec["base"]).to_dict()
    override = spec.get("override", {})
    cells = expand_grid(spec.get("grid", {}))
    return base, override, cells


def cell_config_dict(base: dict, override: dict, cell: dict) -> dict:
    """Deep-copy ``base`` and apply the constant overrides then this cell's assignments."""
    d = copy.deepcopy(base)
    for dotted, value in {**override, **cell}.items():
        _set_dotted(d, dotted, value)
    return d


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_sweep.py <sweep.toml>", file=sys.stderr)
        return 2
    base, override, cells = load_sweep(argv[0])
    from edc.config import load_from_dict  # local: keep module import surface small

    print(f"[edc] sweep '{Path(argv[0]).name}': {len(cells)} cells "
          f"(override={override or '{}'}) -> appending to results/ledger.jsonl")
    for i, cell in enumerate(cells, 1):
        d = cell_config_dict(base, override, cell)
        cfg = load_from_dict(d)
        task_kwargs = d.get("task", {}).get(cfg.run.task, {})
        task = build_task(cfg.run.task, **task_kwargs)
        print(f"\n[edc] cell {i}/{len(cells)}: {cell}")
        row = run_and_append(cfg, task, include_ood=False)
        print(f"[edc] -> run_id={row['run_id']}")
    ledger = REPO_ROOT / "results/ledger.jsonl"
    print(f"\n[edc] sweep complete: {len(cells)} rows appended -> {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
