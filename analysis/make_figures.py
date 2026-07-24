"""Regenerate every paper figure from results/ledger.jsonl. [Phase 3/5]

Deterministic: reads only the ledger (never retrains), calls one function per figure in
``edc.plotting``, writes to analysis/figures/. Each figure records the ledger run_ids it
consumed (provenance). See paper/README.md for the figure->claim map.
"""

from __future__ import annotations

from edc.ledger import read_all


def main() -> int:
    rows = read_all()
    print(f"[figures] {len(rows)} ledger rows available.")
    raise NotImplementedError(
        "Phase 3/5: emit F1..F6 via edc.plotting from the ledger. "
        "Run `make smoke` first to populate results/ledger.jsonl."
    )


if __name__ == "__main__":
    raise SystemExit(main())
