"""Shared figure style + one function per paper figure.

Figures are produced ONLY here and called from ``analysis/make_figures.py``, which reads
the ledger and never re-runs training (invariant 6). matplotlib is an optional dependency
(``--extra plot``); importing this module without it fails only when a plot is requested.

Phase 2+ fills in the concrete F1..F6 functions listed in ``paper/README.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STYLE = {
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def use_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)


# Nonconformity scores rendered in F2, in draw order (geometry emphasised, energy baselines
# muted — the falsification is "geometry beats energy").
_SCORE_STYLE = {
    "geometry": {"color": "#1b6ca8", "lw": 2.4, "label": "geometry (learned)"},
    "rho_basin": {"color": "#5aa469", "lw": 1.6, "label": "1 − ρ_basin (fallback)"},
    "energy_min": {"color": "#b0794a", "lw": 1.4, "label": "energy Eₘᵢₙ (EBT)"},
    "energy_mean": {"color": "#c9a66b", "lw": 1.2, "label": "energy Ē"},
    "energy_std": {"color": "#a05195", "lw": 1.2, "label": "energy spread"},
}


def _selective_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics of the headline ``split == 'selective'`` run for F2/F3 (raises if none).

    Prefers the full-fold run — largest ``n_test`` (E1), ties broken by ledger order — so the
    single-run figures reflect the authoritative experiment, not whatever reduced-fold sweep cell
    happened to land last in the ledger.
    """
    sel = [r for r in rows if r.get("split") == "selective"]
    if not sel:
        raise ValueError("no split='selective' ledger row; run experiments/run_experiment.py first")
    # largest n_test wins; ties broken toward the latest ledger row (freshest full-fold E1).
    headline = max(enumerate(sel), key=lambda iv: (iv[1]["metrics"].get("n_test", 0), iv[0]))[1]
    return headline["metrics"]


def risk_coverage_curve(rows: list[dict[str, Any]], out_path: str) -> str:  # F2
    """Risk-coverage curves per nonconformity score — the core selective-prediction result.

    Reads the sampled ``risk_coverage`` grids stored on the latest selective ledger row (invariant
    6: figures come only from the ledger). Lower curve = better ranker; the geometry line beating
    the energy lines is the visual falsification test.
    """
    import matplotlib.pyplot as plt

    use_style()
    m = _selective_row(rows)
    rc = m["risk_coverage"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for name, style in _SCORE_STYLE.items():
        if name not in rc:
            continue
        aurc = m["aurc"].get(name)
        lbl = f"{style['label']}  (AURC={aurc:.3f})" if aurc is not None else style["label"]
        ax.plot(rc[name]["coverage"], rc[name]["risk"], color=style["color"],
                lw=style["lw"], label=lbl)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("selective risk (error | answered)")
    ax.set_title(f"F2 · risk–coverage  (ID acc {m['accuracy_id']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def k_restart_lift(rows: list[dict[str, Any]], out_path: str) -> str:  # S1 / T2 companion
    """ΔAURC(best energy − geometry) vs number of restarts K — the restart-ablation figure.

    Per-seed points + the seed-mean line, with a dashed y=0 (below ⇒ geometry loses to energy).
    A lift that grows with K is evidence the *restart* geometry (not just a single descent) carries
    the signal. Needs ≥2 distinct K in the ledger; raises otherwise. Data is aggregated in
    ``analysis.aggregate`` but re-read here from the raw rows to keep plotting self-contained.
    """
    import matplotlib.pyplot as plt

    use_style()
    by_k: dict[int, list[float]] = {}
    for r in rows:
        if r.get("split") != "selective":
            continue
        m = r["metrics"]
        k = int(m.get("k_restarts", r["config"]["inference"]["k_restarts"]))
        by_k.setdefault(k, []).append(m["delta_aurc_vs_best_energy"][0])
    if len(by_k) < 2:
        raise ValueError("k_restart_lift needs >=2 distinct K in the ledger (run the K-sweep)")

    ks = sorted(by_k)
    means = [float(np.mean(by_k[k])) for k in ks]
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.axhline(0.0, ls="--", color="0.6", lw=1.0, label="no lift (geometry = energy)")
    for k in ks:
        ax.scatter([k] * len(by_k[k]), by_k[k], color="#1b6ca8", alpha=0.5, s=22, zorder=2)
    ax.plot(ks, means, "-o", color="#1b6ca8", lw=2.0, zorder=3, label="seed mean")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("restarts K  (K=1 ≈ EBT, no basin geometry)")
    ax.set_ylabel("ΔAURC (energy − geometry)")
    ax.set_title("S1 · restart ablation: does geometry lift grow with K?")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def coverage_validity(rows: list[dict[str, Any]], out_path: str) -> str:  # F3
    """Achieved selective risk vs the nominal target α — the calibration-validity check.

    Distribution-free validity means every point sits at or below the diagonal (achieved risk ≤
    target). Reads the ``coverage_validity`` α-sweep from the latest selective ledger row.
    """
    import matplotlib.pyplot as plt

    use_style()
    v = _selective_row(rows)["coverage_validity"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lim = max(v["target"]) * 1.1
    ax.plot([0, lim], [0, lim], ls="--", color="0.6", lw=1.0, label="nominal (achieved = target)")
    ax.scatter(v["target"], v["achieved_risk"], color="#1b6ca8", zorder=3, label="achieved")
    for a, r, c in zip(v["target"], v["achieved_risk"], v["coverage"], strict=True):
        ax.annotate(f"cov {c:.2f}", (a, r), fontsize=6, xytext=(3, 3),
                    textcoords="offset points", color="0.4")
    ax.set_xlabel("target selective risk α")
    ax.set_ylabel("achieved selective risk (test)")
    ax.set_title("F3 · coverage validity (on/below diagonal = valid)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
