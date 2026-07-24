"""End-to-end evaluation orchestration. [Phase 3/4]

Trains (or loads) a reasoner, runs K-restart inference on the eval + calibration splits,
extracts geometry features, computes every nonconformity score (geometry vs the energy/MSP/
MC-dropout/ensemble baselines), calibrates the conformal layer, and returns the metrics dict
that ``experiments/run_experiment.py`` writes to the ledger. Phase-1 ``edc.cli smoke`` covers
the train -> infer -> accuracy path; this module adds the geometry + conformal evaluation.
"""

from __future__ import annotations


def evaluate(cfg, task):
    raise NotImplementedError("Phase 3/4: full geometry + conformal evaluation pipeline.")
