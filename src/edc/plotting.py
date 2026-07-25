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


def _halting_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics of the latest ``split == 'halting'`` run for F4 (raises if none)."""
    sel = [r for r in rows if r.get("split") == "halting"]
    if not sel:
        raise ValueError("no split='halting' ledger row; run experiments/run_halting.py first")
    return sel[-1]["metrics"]


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


_FAMILY_COLOR = {"basin": "#5aa469", "energy": "#b0794a", "curv": "#1b6ca8", "dynamics": "#a05195"}


def _family(name: str) -> str:
    key = name.split("/")[0]
    return "curv" if key == "curv" else key


def _bars_from_hist(h: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(centers, correct_density, incorrect_density, width)`` from a stored histogram."""
    edges = np.asarray(h["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    c = np.asarray(h["correct_counts"], dtype=float)
    i = np.asarray(h["incorrect_counts"], dtype=float)
    c = c / c.sum() if c.sum() else c                     # per-class density (class-balanced view)
    i = i / i.sum() if i.sum() else i
    return centers, c, i, float(edges[1] - edges[0])


def feature_diagnostics(rows: list[dict[str, Any]], out_path: str) -> str:  # F5
    """Per-feature separation + the correct-vs-incorrect distributions of the mechanism features.

    Left: |AUROC−0.5| per feature (distance from chance), coloured by family. Right: overlaid
    correct/incorrect histograms of the most-separating basin and curvature features — the geometry
    signal the thesis rests on. Reads ``feature_diagnostics`` from the headline selective row.
    """
    import matplotlib.pyplot as plt

    fd = _selective_row(rows).get("feature_diagnostics")
    if fd is None:
        raise ValueError("no feature_diagnostics on the selective row; re-run run_experiment.py")
    use_style()
    names = fd["names"]
    sep = {n: abs(fd["auroc"][n] - 0.5) for n in names}
    order = sorted(names, key=lambda n: sep[n])            # ascending -> strongest at top

    def top(family: str) -> str:
        fam = [n for n in names if _family(n) == family]
        return max(fam, key=lambda n: sep[n]) if fam else names[0]

    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})

    ax0.barh(range(len(order)), [sep[n] for n in order],
             color=[_FAMILY_COLOR[_family(n)] for n in order])
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels(order, fontsize=7)
    ax0.set_xlabel("|AUROC − 0.5|  (correct-vs-incorrect separation)")
    ax0.set_title("F5 · which geometry features separate")

    for ax, family in ((ax1, "basin"), (ax2, "curv")):
        name = top(family)
        centers, c, i, w = _bars_from_hist(fd["hist"][name])
        ax.bar(centers, c, width=w, color="#1b6ca8", alpha=0.6, label="correct")
        ax.bar(centers, i, width=w, color="#d1495b", alpha=0.6, label="incorrect")
        ax.set_title(f"{name}\n(AUROC {fd['auroc'][name]:.2f})", fontsize=9)
        ax.set_xlabel("feature value")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def ood_stress(rows: list[dict[str, Any]], out_path: str) -> str:  # F6
    """Selective-risk validity in-distribution vs under distribution shift.

    The LTT threshold is calibrated ID; achieved risk vs target α is plotted for the ID test fold
    (on/below the diagonal = valid) and the OOD test fold (above the diagonal = the guarantee breaks
    because exchangeability is violated). This is the argument for abstention/routing in critical
    systems. Reads ``ood_validity`` from the headline selective row.
    """
    import matplotlib.pyplot as plt

    v = _selective_row(rows).get("ood_validity")
    if v is None:
        raise ValueError("no ood_validity on the selective row; re-run with include_ood=True")
    use_style()
    lim = max(v["target"]) * 1.1
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, lim], [0, lim], ls="--", color="0.6", lw=1.0, label="valid (achieved ≤ target)")
    ax.plot(v["target"], v["id_risk"], "-o", color="#1b6ca8", lw=1.8, label="in-distribution")
    ax.plot(v["target"], v["ood_risk"], "-s", color="#d1495b", lw=1.8, label="out-of-distribution")
    ax.set_xlabel("target selective risk α")
    ax.set_ylabel("achieved selective risk (test)")
    ax.set_title("F6 · guarantee holds ID, breaks under shift")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, max(1.0, max(v["ood_risk"]) * 1.05) if v["ood_risk"] else 1.0)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def halting_pareto(rows: list[dict[str, Any]], out_path: str) -> str:  # F4
    """Compute-vs-accuracy Pareto under adaptive halting.

    Sweeps the agreement threshold τ: end-task accuracy (y) vs mean fraction of the step budget
    used (x). Marks the CRC-chosen operating point (guaranteed halting risk ≤ α) and the full-budget
    reference. Reads the latest ``split=='halting'`` ledger row (invariant 6).
    """
    import matplotlib.pyplot as plt

    use_style()
    m = _halting_row(rows)
    s = m["tau_sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(s["compute_used"], s["accuracy"], "-o", color="#1b6ca8", lw=1.8, ms=4,
            label="halting τ-sweep", zorder=2)
    ax.axhline(m["full_accuracy"], ls="--", color="0.6", lw=1.0,
               label=f"full-budget acc {m['full_accuracy']:.2f}")
    if m["tau_hat"] is not None:
        lbl = (f"CRC τ̂={m['tau_hat']:.2f} (risk {m['halting_risk']:.2f} ≤ α={m['alpha']}), "
               f"{m['compute_saved']:.0%} compute saved")
        ax.scatter([m["compute_used"]], [m["halted_accuracy"]], color="#d1495b", s=70, zorder=3,
                   label=lbl)
    ax.set_xlabel("compute used (fraction of step budget)")
    ax.set_ylabel("end-task accuracy")
    ax.set_title("F4 · adaptive halting: accuracy vs compute")
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
