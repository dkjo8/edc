"""Sweep grid expansion and dotted-key overrides — pure config manipulation, no training.

Offline + CPU-only: exercises only the parsing/expansion helpers of experiments/run_sweep.py
(the per-cell training is covered by the slow pipeline test).
"""

import run_sweep


def test_set_dotted_nested():
    d = {"inference": {"k_restarts": 8}}
    run_sweep._set_dotted(d, "inference.k_restarts", 16)
    run_sweep._set_dotted(d, "run.seed", 3)          # creates the intermediate dict
    assert d["inference"]["k_restarts"] == 16
    assert d["run"]["seed"] == 3


def test_expand_grid_cartesian():
    cells = run_sweep.expand_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(cells) == 6
    assert {"a": 1, "b": 10} in cells and {"a": 2, "b": 30} in cells
    assert run_sweep.expand_grid({}) == [{}]          # empty grid -> one no-op cell


def test_cell_config_applies_override_and_cell():
    base = {"inference": {"k_restarts": 8, "steps": 50}, "eval": {"n_eval": 1500},
            "run": {"seed": 0}}
    override = {"eval.n_eval": 600}
    cell = {"inference.k_restarts": 4, "run.seed": 2}
    d = run_sweep.cell_config_dict(base, override, cell)
    assert d["eval"]["n_eval"] == 600                 # override applied
    assert d["inference"]["k_restarts"] == 4 and d["run"]["seed"] == 2   # cell applied
    assert d["inference"]["steps"] == 50              # untouched key preserved
    assert base["inference"]["k_restarts"] == 8       # base not mutated (deep-copied)


def test_load_sweep_reads_real_config():
    base, override, cells = run_sweep.load_sweep("configs/sweeps/k_restarts.toml")
    assert len(cells) == 25                           # 5 K x 5 seeds
    assert override["eval.n_eval"] == 600
    assert base["task"]["arithmetic"]["n_operands"] == 6   # base experiment config resolved
