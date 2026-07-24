"""Regenerate LaTeX tables (T1..T3) from results/ledger.jsonl. [Phase 3/5]

Main-results, ablation, and reproducibility-appendix tables, emitted to paper/. Reads only the
ledger. See paper/README.md for the table->claim map.
"""

from __future__ import annotations

from edc.ledger import read_all


def main() -> int:
    rows = read_all()
    print(f"[tables] {len(rows)} ledger rows available.")
    raise NotImplementedError("Phase 3/5: emit T1..T3 LaTeX from the ledger.")


if __name__ == "__main__":
    raise SystemExit(main())
