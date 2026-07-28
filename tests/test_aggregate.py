"""Ledger aggregation across seeds: grouping by K + the multi-seed falsification summary.

Offline + CPU-only, on hand-built synthetic ledger rows (no training).
"""

import aggregate


def _row(k, seed, delta, lo, geo=0.08, best=0.14, task="arithmetic", baseline_delta=None,
         geom_adds=None):
    m = {
        "k_restarts": k, "accuracy_id": 0.8, "base_error": 0.2,
        "best_energy_baseline": "energy_min",
        "aurc": {"geometry": geo, "energy_min": best},
        "delta_aurc_vs_best_energy": [delta, lo, delta + 0.02],
        "ltt": {"coverage": 0.7, "selective_risk": 0.07, "abstain_rate": 0.3},
    }
    if baseline_delta is not None:                          # Phase 4e field (older rows omit it)
        bd, blo = baseline_delta
        m["best_baseline"] = "temp_msp"
        m["delta_aurc_vs_best_baseline"] = [bd, blo, bd + 0.02]
        m["feature_ablation"] = {                            # Phase 4f field
            "full": geo, "drop_basin": geo + 0.03, "drop_energy": geo + 0.005,
            "drop_curv": geo + 0.001, "drop_dynamics": geo + 0.002,
        }
    if geom_adds is not None:                               # Phase 4j field
        ga, glo = geom_adds
        m["delta_aurc_geom_adds"] = [ga, glo, ga + 0.02]
    return {
        "run_id": f"{task[:3]}{k}_{seed}", "seed": seed, "split": "selective", "task": task,
        "config_hash": f"{task}h{k}_{seed}", "git_sha": "abc1234",
        "env": {"python": "3.12", "jax": "0.11", "jax_backend": "cpu"},
        "config": {"inference": {"k_restarts": k}},
        "metrics": m,
    }


def test_by_k_groups_and_dedups():
    rows = [_row(1, 0, 0.0, -0.01), _row(1, 1, 0.0, -0.01), _row(4, 0, 0.06, 0.05)]
    # a re-run of (K=4, seed 0) with the SAME config_hash must replace, not duplicate
    rows.append(_row(4, 0, 0.07, 0.06))
    grouped = aggregate.by_k(rows)
    assert set(grouped) == {1, 4}
    assert len(grouped[1]) == 2 and len(grouped[4]) == 1
    assert grouped[4][0]["metrics"]["delta_aurc_vs_best_energy"][0] == 0.07   # latest kept


def test_aggregate_cell_verdict():
    # K=1: no lift (CI includes 0). K=16: clear lift on all seeds.
    k1 = [_row(1, s, 0.001, -0.01) for s in range(5)]
    a1 = aggregate.aggregate_cell(k1)
    assert a1["seeds_ci_excludes_0"] == 0 and a1["geometry_wins_all"] is False
    assert a1["n_seeds"] == 5 and a1["k_restarts"] == 1

    k16 = [_row(16, s, 0.09, 0.07, geo=0.05) for s in range(5)]
    a16 = aggregate.aggregate_cell(k16)
    assert a16["seeds_ci_excludes_0"] == 5 and a16["geometry_wins_all"] is True
    assert a16["delta_aurc"][0] > a1["delta_aurc"][0]           # lift grows with K
    assert a16["aurc_geometry"][0] < a16["aurc_best_energy"][0]  # geometry lower AURC = better


def test_baseline_delta_aggregation_and_fallback():
    # Rows carrying the Phase-4e field aggregate its own value + seeds-win count.
    withbl = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03)) for s in range(4)]
    a = aggregate.aggregate_cell(withbl)
    assert a["delta_aurc_baseline"][0] == 0.05 and a["seeds_ci_excludes_0_baseline"] == 4
    assert a["delta_aurc"][0] == 0.09                       # energy ΔAURC still separate
    assert "temp_msp" in a["best_baseline_names"]

    # Older rows without the field fall back to the energy ΔAURC (no crash).
    old = [_row(12, s, 0.09, 0.07) for s in range(3)]
    a_old = aggregate.aggregate_cell(old)
    assert a_old["delta_aurc_baseline"] == a_old["delta_aurc"]
    assert a_old["feature_ablation"] == {}                  # absent -> empty, no crash
    assert a_old["has_geom_adds"] is False                  # Phase 4j field absent -> flagged


def test_geom_adds_complementarity_aggregation():
    # rows carrying the Phase-4j complementarity field aggregate it + count seeds that clear 0.
    rows = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03), geom_adds=(0.02, 0.008))
            for s in range(5)]
    a = aggregate.aggregate_cell(rows)
    assert a["has_geom_adds"] is True
    assert a["delta_aurc_geom_adds"][0] == 0.02 and a["seeds_ci_excludes_0_geom_adds"] == 5
    # a tie (CI includes 0) is not counted as geometry adding
    tie = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03), geom_adds=(0.001, -0.004))
           for s in range(5)]
    assert aggregate.aggregate_cell(tie)["seeds_ci_excludes_0_geom_adds"] == 0


def test_feature_ablation_aggregates():
    rows = [_row(12, s, 0.09, 0.07, geo=0.08, baseline_delta=(0.05, 0.03)) for s in range(4)]
    fa = aggregate.aggregate_cell(rows)["feature_ablation"]
    assert set(fa) == {"full", "drop_basin", "drop_energy", "drop_curv", "drop_dynamics"}
    assert fa["full"][0] == 0.08 and fa["drop_basin"][0] == 0.11   # mean AURC per subset
    assert fa["full"][1] == 0.0                                    # zero std (identical seeds)


def test_task_filter_no_cross_contamination():
    # Two tasks share K=12 in the ledger; aggregation must not mix them.
    rows = [_row(12, s, 0.08, 0.06, task="arithmetic") for s in range(3)]
    rows += [_row(12, s, 0.02, -0.02, task="graph_planning") for s in range(4)]
    assert aggregate.tasks_present(rows) == ["arithmetic", "graph_planning"]

    ar = aggregate.aggregate_by_k(rows, task="arithmetic")[12]
    gr = aggregate.aggregate_by_k(rows, task="graph_planning")[12]
    assert ar["n_seeds"] == 3 and gr["n_seeds"] == 4
    assert ar["task"] == "arithmetic" and gr["task"] == "graph_planning"
    assert ar["delta_aurc"][0] != gr["delta_aurc"][0]           # distinct aggregates
    # unfiltered mixes both (7 rows at K=12) — the bug the filter prevents
    assert aggregate.aggregate_by_k(rows)[12]["n_seeds"] == 7
