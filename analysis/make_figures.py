"""Regenerate every paper figure from results/ledger.jsonl. [Phase 3/5]

Deterministic: reads only the ledger (never retrains), calls one function per figure in
``edc.plotting``, writes to analysis/figures/. Each figure records the ledger run_ids it
consumed (provenance). See paper/README.md for the figure->claim map.
"""

from __future__ import annotations

from pathlib import Path

from edc import plotting
from edc.ledger import read_all

FIG_DIR = Path(__file__).resolve().parent / "figures"

# (filename, plotting fn). F1 (schematic) is TikZ; F4/F6 (halting, OOD stress) are Phase 4/5.
_FIGURES = [
    ("F2_risk_coverage.png", plotting.risk_coverage_curve),
    ("F3_coverage_validity.png", plotting.coverage_validity),
    ("F4_halting_pareto.png", plotting.halting_pareto),
    ("S1_k_restart_lift.png", plotting.k_restart_lift),
]


def main() -> int:
    rows = read_all()
    print(f"[figures] {len(rows)} ledger rows available.")
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
