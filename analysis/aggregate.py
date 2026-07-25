"""Aggregate ``split='selective'`` ledger rows across seeds for the tables/figures. [Phase 4]

Pure NumPy over already-computed per-seed metrics (invariant 6: everything derives from the
ledger; no re-training, no re-bootstrapping). The multi-seed ΔAURC is summarised from each row's
own point estimate + 95% CI — we report the mean±std over seeds and how many seeds' CIs exclude 0,
rather than pooling raw scores across seeds (which would break the per-run exchangeability).
"""

from __future__ import annotations

import numpy as np

from edc.ledger import read_all


def selective_rows(rows: list[dict] | None = None) -> list[dict]:
    """Latest ``split=='selective'`` row per ``(config_hash, seed)`` (cf. latest_per_config)."""
    if rows is None:
        rows = read_all()
    latest: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r.get("split") == "selective":
            latest[(r["config_hash"], r["seed"])] = r  # later rows win
    return list(latest.values())


def _k(row: dict) -> int:
    m = row["metrics"]
    return int(m.get("k_restarts", row["config"]["inference"]["k_restarts"]))


def by_k(rows: list[dict] | None = None) -> dict[int, list[dict]]:
    """Group selective rows by ``k_restarts`` (ascending)."""
    out: dict[int, list[dict]] = {}
    for r in selective_rows(rows):
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
    return {
        "k_restarts": _k(rows_for_k[0]),
        "n_seeds": n,
        "seeds": sorted(r["seed"] for r in rows_for_k),
        "run_ids": [r["run_id"] for r in rows_for_k],
        "accuracy_id": _ms([m["accuracy_id"] for m in ms]),
        "aurc_geometry": (geo_mean, geo_std),
        "aurc_best_energy": (best_mean, best_std),
        "best_energy_names": sorted({m["best_energy_baseline"] for m in ms}),
        "delta_aurc": (delta_mean, delta_std),
        "seeds_ci_excludes_0": ci_excludes_0,
        "geometry_wins_all": bool(ci_excludes_0 == n and delta_mean > 0),
        "ltt_coverage": _ms([m["ltt"]["coverage"] for m in ms]),
        "ltt_selective_risk": _ms([m["ltt"]["selective_risk"] for m in ms]),
        "ltt_abstain_rate": _ms([m["ltt"]["abstain_rate"] for m in ms]),
    }


def aggregate_by_k(rows: list[dict] | None = None) -> dict[int, dict]:
    """``{K: aggregate_cell(...)}`` for every K present in the ledger."""
    return {k: aggregate_cell(v) for k, v in by_k(rows).items()}
