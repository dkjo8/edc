"""F5/F6 plotting: renders a PNG from a synthetic ledger row and skips cleanly when the block is
absent. Offline + CPU-only; skipped if matplotlib (the optional plot extra) is unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from edc import plotting  # noqa: E402


def _row_with_blocks():
    names = ["basin/rho", "basin/entropy", "curv/lmax_best", "energy/min", "dynamics/steps"]
    rng = np.random.default_rng(0)
    hist = {n: {"edges": np.linspace(0, 1, 21).tolist(),
                "correct_counts": rng.integers(0, 30, 20).tolist(),
                "incorrect_counts": rng.integers(0, 10, 20).tolist()} for n in names}
    metrics = {
        "n_test": 500,
        "feature_diagnostics": {
            "names": names,
            "auroc": {"basin/rho": 0.78, "basin/entropy": 0.30, "curv/lmax_best": 0.66,
                      "energy/min": 0.55, "dynamics/steps": 0.50},
            "hist": hist,
        },
        "ood_validity": {
            "target": [0.05, 0.1, 0.2, 0.3],
            "id_risk": [0.03, 0.08, 0.15, 0.22],       # ID stays on/below diagonal
            "ood_risk": [0.4, 0.5, 0.6, 0.7],          # OOD breaks above diagonal
            "id_coverage": [0.2, 0.5, 0.8, 0.95],
            "ood_coverage": [0.3, 0.6, 0.85, 0.97],
        },
    }
    return [{"split": "selective", "metrics": metrics}]


def test_f5_f6_render(tmp_path):
    rows = _row_with_blocks()
    f5 = plotting.feature_diagnostics(rows, str(tmp_path / "f5.png"))
    f6 = plotting.ood_stress(rows, str(tmp_path / "f6.png"))
    assert (tmp_path / "f5.png").stat().st_size > 0
    assert (tmp_path / "f6.png").stat().st_size > 0
    assert f5.endswith("f5.png") and f6.endswith("f6.png")


def test_missing_blocks_raise_valueerror(tmp_path):
    rows = [{"split": "selective", "metrics": {"n_test": 10}}]   # no F5/F6 blocks
    with pytest.raises(ValueError):
        plotting.feature_diagnostics(rows, str(tmp_path / "f5.png"))
    with pytest.raises(ValueError):
        plotting.ood_stress(rows, str(tmp_path / "f6.png"))
