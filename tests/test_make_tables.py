"""LaTeX table generation smoke test: T1/T2/T3 are well-formed from synthetic ledger rows.

Offline + CPU-only. Checks structure (booktabs rules, labels, per-task T1, per-K T2) without
asserting exact numbers.
"""

import aggregate
import make_tables
from test_aggregate import _row


def _rows():
    rows = []
    for k in (1, 4, 16):                                     # arithmetic K-sweep
        lift = 0.0 if k == 1 else 0.05
        for s in range(3):
            rows.append(_row(k, s, lift, lift - 0.01, geo=0.1 - lift, best=0.14))
    for s in range(3):                                       # a second task at one K
        rows.append(_row(12, s, 0.03, 0.01, geo=0.09, best=0.13, task="graph_planning"))
    return rows


def test_tables_well_formed():
    rows = _rows()

    t1 = make_tables._t1(rows)                               # per-task
    t2 = make_tables._t2(aggregate.aggregate_by_k(rows, task="arithmetic"))
    t3 = make_tables._t3(rows)

    for tex in (t1, t2, t3):
        assert r"\toprule" in tex and r"\bottomrule" in tex and r"\begin{tabular}" in tex

    assert r"\label{tab:main}" in t1
    assert r"\label{tab:kablation}" in t2
    # T1 has one row per task (arithmetic + graph_planning)
    t1_body = t1.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert t1_body.count(r"\\") == 2
    assert "graph" in t1 and "arithmetic" in t1
    # T2 has one data row per arithmetic K (1, 4, 16)
    t2_body = t2.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert t2_body.count(r"\\") == 3
    # T3 lists every run
    assert t3.count(r"\\") >= len(rows)


def test_delta_pm_formatting():
    assert make_tables._pm((0.0736, 0.012)) == r"$0.074 \pm 0.012$"
