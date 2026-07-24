"""Run a single named experiment from a config and append a ledger row. [Phase 4]

    PYTHONPATH=src python experiments/run_experiment.py configs/experiments/<name>.toml

Phase 1 uses ``edc.cli smoke`` for the train->infer->accuracy path. This runner adds the full
geometry + conformal evaluation (``edc.eval.evaluate``) once Phase 3 lands.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    raise NotImplementedError(
        "Phase 4: load config -> edc.eval.evaluate -> edc.ledger.append. "
        "For now use: PYTHONPATH=src python -m edc.cli smoke --config configs/smoke.toml"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
