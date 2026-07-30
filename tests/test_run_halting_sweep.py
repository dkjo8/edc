"""run_halting_sweep expands a seed grid and appends split='halting' rows via evaluate_halting.

Offline + CPU-only and FAST: ``evaluate_halting`` and the ledger ``append`` are monkeypatched, so
no reasoner is trained and the real ledger is never written — we only check the runner's plumbing
(grid expansion, one halting row per cell, correct split/seed).
"""

import run_halting_sweep


def _fake_metrics():
    return {"tau_hat": 0.9, "compute_saved": 0.5, "halting_risk": 0.02,
            "risk_within_budget": True, "alpha": 0.1}


def test_run_halting_sweep_appends_one_halting_row_per_cell(tmp_path, monkeypatch):
    sweep = tmp_path / "smoke_halting.toml"
    sweep.write_text(
        '[sweep]\nbase = "configs/smoke.toml"\n[sweep.grid]\n"run.seed" = [0, 1, 2]\n')

    calls = {"eval": 0}
    captured = []

    def fake_eval(cfg, task):
        calls["eval"] += 1
        return _fake_metrics()

    def fake_append(record):
        row = record.to_row() if hasattr(record, "to_row") else record
        captured.append(row)
        return row

    monkeypatch.setattr(run_halting_sweep, "evaluate_halting", fake_eval)
    monkeypatch.setattr(run_halting_sweep, "append", fake_append)

    rc = run_halting_sweep.main([str(sweep)])
    assert rc == 0
    assert calls["eval"] == 3                              # one evaluate_halting per seed cell
    assert len(captured) == 3
    assert all(r["split"] == "halting" for r in captured)
    assert sorted(r["seed"] for r in captured) == [0, 1, 2]
