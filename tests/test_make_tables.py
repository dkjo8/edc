"""LaTeX table generation smoke test: T1/T2/T3 are well-formed from synthetic ledger rows.

Offline + CPU-only. Checks structure (booktabs rules, labels, per-task T1, per-K T2) without
asserting exact numbers.
"""

import aggregate
import make_tables
from test_aggregate import _hrow, _row


def _rows():
    rows = []
    for k in (1, 4, 16):                                     # arithmetic K-sweep
        lift = 0.0 if k == 1 else 0.05
        for s in range(3):
            rows.append(_row(k, s, lift, lift - 0.01, geo=0.1 - lift, best=0.14,
                             baseline_delta=(lift - 0.02, lift - 0.03)))
    for s in range(3):                                       # a second task at one K
        rows.append(_row(12, s, 0.03, 0.01, geo=0.09, best=0.13, task="graph_planning",
                         baseline_delta=(0.01, -0.01)))
    return rows


def test_tables_well_formed():
    rows = _rows()

    t1 = make_tables._t1(rows)                               # per-task
    t2 = make_tables._t2(aggregate.aggregate_by_k(rows, task="arithmetic"))
    t2b = make_tables._t2b(rows)                             # feature ablation
    t3 = make_tables._t3(rows)

    for tex in (t1, t2, t2b, t3):
        assert r"\toprule" in tex and r"\bottomrule" in tex and r"\begin{tabular}" in tex

    assert r"\label{tab:main}" in t1
    assert r"\label{tab:kablation}" in t2
    assert r"\label{tab:ablation}" in t2b and "drop\\_energy" in t2b
    # T1 has one row per task (arithmetic + graph_planning)
    t1_body = t1.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert t1_body.count(r"\\") == 2
    assert "graph" in t1 and "arithmetic" in t1
    # T2 has one data row per arithmetic K (1, 4, 16)
    t2_body = t2.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert t2_body.count(r"\\") == 3
    # T3 lists every run
    assert t3.count(r"\\") >= len(rows)


def test_t6_halting_and_t7_ood():
    # T6 from split="halting" rows; T7 from include_ood selective rows (Phase 4l).
    hrows = [_hrow(s, 0.55, 0.02, task="arithmetic") for s in range(5)]
    t6 = make_tables._t6(hrows)
    assert r"\label{tab:halting}" in t6
    assert t6.split(r"\midrule")[1].split(r"\bottomrule")[0].count(r"\\") == 1   # one task row
    assert "arithmetic" in t6

    ood_rows = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03),
                     ood=(0.4, 0.5, 0.8, False)) for s in range(5)]
    t7 = make_tables._t7(ood_rows)
    assert r"\label{tab:oodstress}" in t7
    assert "arithmetic" in t7 and r"\toprule" in t7

    # both are gated: absent data -> empty body
    assert make_tables._t6([]).split(r"\midrule")[1].split(r"\bottomrule")[0].strip() == ""
    no_ood = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03)) for s in range(3)]
    assert make_tables._t7(no_ood).split(r"\midrule")[1].split(r"\bottomrule")[0].strip() == ""


def test_t8_ensemble():
    # T8 from deep-ensemble rows (Phase 4m); gated when absent.
    rows = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03),
                 ensemble=(0.02, 0.005, 0.10, 0.82, 5)) for s in range(5)]
    t8 = make_tables._t8(rows)
    assert r"\label{tab:ensemble}" in t8
    assert t8.split(r"\midrule")[1].split(r"\bottomrule")[0].count(r"\\") == 1   # one cell row
    assert "arithmetic" in t8

    no_ens = [_row(12, s, 0.09, 0.07, baseline_delta=(0.05, 0.03)) for s in range(3)]
    assert make_tables._t8(no_ens).split(r"\midrule")[1].split(r"\bottomrule")[0].strip() == ""


def test_delta_pm_formatting():
    assert make_tables._pm((0.0736, 0.012)) == r"$0.074 \pm 0.012$"
