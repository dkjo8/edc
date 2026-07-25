"""Shared figure style + one function per paper figure.

Figures are produced ONLY here and called from ``analysis/make_figures.py``, which reads
the ledger and never re-runs training (invariant 6). matplotlib is an optional dependency
(``--extra plot``); importing this module without it fails only when a plot is requested.

Phase 2+ fills in the concrete F1..F6 functions listed in ``paper/README.md``.
"""

from __future__ import annotations

from typing import Any

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
    """Latest ``split == 'selective'`` ledger row's metrics (raises if none)."""
    sel = [r for r in rows if r.get("split") == "selective"]
    if not sel:
        raise ValueError("no split='selective' ledger row; run experiments/run_experiment.py first")
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
