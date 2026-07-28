"""Regenerate every paper figure from results/ledger.jsonl. [Phase 3/5]

Deterministic: reads only the ledger (never retrains), calls one function per figure in
``edc.plotting``, writes to analysis/figures/. Each figure records the ledger run_ids it
consumed (provenance). See paper/README.md for the figure->claim map.
"""

from __future__ import annotations

from pathlib import Path

from edc import plotting
from edc.ledger import read_all

try:  # sibling import when run as a script (analysis/ on sys.path)
    from aggregate import feature_set_of
except ImportError:  # pragma: no cover
    from analysis.aggregate import feature_set_of

FIG_DIR = Path(__file__).resolve().parent / "figures"

# (filename, plotting fn). F1 (schematic) is TikZ; F4/F6 (halting, OOD stress) are Phase 4/5.
_FIGURES = [
    ("F2_risk_coverage.png", plotting.risk_coverage_curve),
    ("F3_coverage_validity.png", plotting.coverage_validity),
    ("F4_halting_pareto.png", plotting.halting_pareto),
    ("F5_feature_diagnostics.png", plotting.feature_diagnostics),
    ("F6_ood_stress.png", plotting.ood_stress),
    ("S1_k_restart_lift.png", plotting.k_restart_lift),
]


def main() -> int:
    # Canonical figures are all base-feature results; exclude Phase-4k richer rows so they never
    # dilute the pooled figures (e.g. S1's K=12 point). The richer verdict lives in table T5.
    rows = [r for r in read_all() if feature_set_of(r) == "base"]
    print(f"[figures] {len(rows)} base-feature ledger rows available.")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for fname, fn in _FIGURES:
        out = FIG_DIR / fname
        try:
            fn(rows, str(out))
        except (ValueError, NotImplementedError, ImportError) as e:
            print(f"[figures] skip {fname}: {e}")
            continue
        n_sel = sum(1 for r in rows if r.get("split") == "selective")
        print(f"[figures] wrote {out}  (from {n_sel} selective rows)")
        n_ok += 1
    print(f"[figures] {n_ok}/{len(_FIGURES)} figures generated -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
