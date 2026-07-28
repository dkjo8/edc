"""Aggregate ``split='selective'`` ledger rows across seeds for the tables/figures. [Phase 4]

Pure NumPy over already-computed per-seed metrics (invariant 6: everything derives from the
ledger; no re-training, no re-bootstrapping). The multi-seed ΔAURC is summarised from each row's
own point estimate + 95% CI — we report the mean±std over seeds and how many seeds' CIs exclude 0,
rather than pooling raw scores across seeds (which would break the per-run exchangeability).
"""

from __future__ import annotations

import numpy as np

from edc.ledger import read_all


def objective_of(row: dict) -> str:
    """Training objective behind a row: ``basin_center`` (default) or ``ired``."""
    return row["metrics"].get("objective") or row["config"].get("train", {}).get(
        "objective", "basin_center")


def selective_rows(rows: list[dict] | None = None, task: str | None = None,
                   objective: str | None = None) -> list[dict]:
    """Latest ``split=='selective'`` row per ``(config_hash, seed)`` (cf. latest_per_config).

    ``task`` filters to one reasoning family and ``objective`` to one reasoner (``basin_center`` vs
    ``ired``), so multi-task / multi-reasoner ledgers do not cross-contaminate per-K aggregates.
    """
    if rows is None:
        rows = read_all()
    latest: dict[tuple[str, int], dict] = {}
    for r in rows:
        if (r.get("split") == "selective" and (task is None or r.get("task") == task)
                and (objective is None or objective_of(r) == objective)):
            latest[(r["config_hash"], r["seed"])] = r  # later rows win
    return list(latest.values())


def tasks_present(rows: list[dict] | None = None) -> list[str]:
    """Sorted distinct task names among selective rows."""
    return sorted({r.get("task") for r in selective_rows(rows)} - {None})


def _k(row: dict) -> int:
    m = row["metrics"]
    return int(m.get("k_restarts", row["config"]["inference"]["k_restarts"]))


def by_k(rows: list[dict] | None = None, task: str | None = None) -> dict[int, list[dict]]:
    """Group selective rows by ``k_restarts`` (ascending), optionally filtered to one ``task``."""
    out: dict[int, list[dict]] = {}
    for r in selective_rows(rows, task=task):
        out.setdefault(_k(r), []).append(r)
    return dict(sorted(out.items()))


def _ms(vals) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def aggregate_cell(rows_for_k: list[dict]) -> dict:
    """Aggregate one K's per-seed rows into mean/std stats + the multi-seed verdict."""
    ms = [r["metrics"] for r in rows_for_k]
    dvb = [m["delta_aurc_vs_best_energy"] for m in ms]  # each [delta, lo, hi]
    n = len(ms)
    delta_mean, delta_std = _ms([d[0] for d in dvb])
    ci_excludes_0 = sum(1 for d in dvb if d[1] > 0.0)  # lo > 0 => positive & CI clears 0
    geo_mean, geo_std = _ms([m["aurc"]["geometry"] for m in ms])
    best_mean, best_std = _ms([m["aurc"][m["best_energy_baseline"]] for m in ms])
    # vs the strongest of ALL baselines (energy + softmax); older rows fall back to energy.
    dvbl = [m.get("delta_aurc_vs_best_baseline", m["delta_aurc_vs_best_energy"]) for m in ms]
    dbl_mean, dbl_std = _ms([d[0] for d in dvbl])
    bl_ci = sum(1 for d in dvbl if d[1] > 0.0)
    # Complementarity (Phase 4j): does geometry add over a learned softmax? (pre-4j rows lack it)
    has_add = all("delta_aurc_geom_adds" in m for m in ms)
    dadd = [m["delta_aurc_geom_adds"] for m in ms] if has_add else []
    add_mean, add_std = _ms([d[0] for d in dadd]) if has_add else (float("nan"), float("nan"))
    add_ci = sum(1 for d in dadd if d[1] > 0.0)
    return {
        "task": rows_for_k[0].get("task"),
        "k_restarts": _k(rows_for_k[0]),
        "n_seeds": n,
        "seeds": sorted(r["seed"] for r in rows_for_k),
        "run_ids": [r["run_id"] for r in rows_for_k],
        "accuracy_id": _ms([m["accuracy_id"] for m in ms]),
        "aurc_geometry": (geo_mean, geo_std),
        "aurc_best_energy": (best_mean, best_std),
        "best_energy_names": sorted({m["best_energy_baseline"] for m in ms}),
        "best_baseline_names": sorted(
            {m.get("best_baseline", m["best_energy_baseline"]) for m in ms}),
        "delta_aurc": (delta_mean, delta_std),
        "seeds_ci_excludes_0": ci_excludes_0,
        "geometry_wins_all": bool(ci_excludes_0 == n and delta_mean > 0),
        "delta_aurc_baseline": (dbl_mean, dbl_std),
        "seeds_ci_excludes_0_baseline": bl_ci,
        "delta_aurc_geom_adds": (add_mean, add_std),
        "seeds_ci_excludes_0_geom_adds": add_ci,
        "has_geom_adds": has_add,
        # True only when every row genuinely carried the Phase-4e baseline field (not a fallback).
        "has_baseline": all("delta_aurc_vs_best_baseline" in m for m in ms),
        # Feature-group leave-one-out ablation (Phase 4f), mean/std per subset over seeds.
        "feature_ablation": (
            {k: _ms([m["feature_ablation"][k] for m in ms])
             for k in ms[0]["feature_ablation"]}
            if all("feature_ablation" in m for m in ms) else {}),
        "ltt_coverage": _ms([m["ltt"]["coverage"] for m in ms]),
        "ltt_selective_risk": _ms([m["ltt"]["selective_risk"] for m in ms]),
        "ltt_abstain_rate": _ms([m["ltt"]["abstain_rate"] for m in ms]),
    }


def aggregate_by_k(rows: list[dict] | None = None, task: str | None = None) -> dict[int, dict]:
    """``{K: aggregate_cell(...)}`` for every K present (optionally filtered to one ``task``)."""
    return {k: aggregate_cell(v) for k, v in by_k(rows, task=task).items()}


def headline_cell(rows: list[dict] | None = None, task: str | None = None,
                  objective: str = "basin_center") -> dict:
    """Aggregate the headline seed-sweep for one task + reasoner (largest single-config seed group).

    T1 uses the canonical ``basin_center`` reasoner (pass ``objective="ired"`` to report the learned
    landscape separately). Prefers rows carrying the Phase-4e baseline field, then picks the config
    group (K, n_test) with the most seeds — the multi-seed sweep — so the cell is not polluted by
    mixing fold sizes / reasoners that happen to share a K value.
    """
    sel = selective_rows(rows, task=task, objective=objective) or selective_rows(rows, task=task)
    with_bl = [r for r in sel if "delta_aurc_vs_best_baseline" in r["metrics"]]
    pool = with_bl or sel
    # Group by the experiment config *excluding seed* — (K, n_test) separates a seed-sweep (many
    # seeds, one fold size) from one-off full-fold runs at the same K.
    groups: dict[tuple, list[dict]] = {}
    for r in pool:
        groups.setdefault((_k(r), r["metrics"].get("n_test")), []).append(r)

    def _dedup_by_seed(group: list[dict]) -> list[dict]:
        # A config re-run after new config fields were added gets a fresh config_hash, so old + new
        # rows for one seed both survive `selective_rows`; keep the latest per seed by timestamp.
        by_seed: dict[int, dict] = {}
        for r in sorted(group, key=lambda r: r.get("timestamp", "")):
            by_seed[r["seed"]] = r
        return list(by_seed.values())

    cells = [_dedup_by_seed(g) for g in groups.values()]
    # Prefer a cell whose rows carry the newest complementarity metric, then more seeds, then most
    # recent — so the headline is the latest full seed-sweep, not a stale or partial one.
    def _key(cell: list[dict]) -> tuple:
        has_add = all("delta_aurc_geom_adds" in r["metrics"] for r in cell)
        return (has_add, len(cell), max(r.get("timestamp", "") for r in cell))

    return aggregate_cell(max(cells, key=_key))
