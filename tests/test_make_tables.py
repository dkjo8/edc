"""LaTeX table generation smoke test: T1/T2/T3 are well-formed from synthetic ledger rows.

Offline + CPU-only. Checks structure (booktabs rules, labels, one T2 row per K) without asserting
exact numbers.
"""

import aggregate
import make_tables
from test_aggregate import _row


def _rows():
    rows = []
    for k in (1, 4, 16):
        lift = 0.0 if k == 1 else 0.05
        for s in range(3):
            rows.append(_row(k, s, lift, lift - 0.01, geo=0.1 - lift, best=0.14))
    return rows


def test_tables_well_formed():
    rows = _rows()
    agg = aggregate.aggregate_by_k(rows)

    t1 = make_tables._t1(agg, primary_k=max(agg))
    t2 = make_tables._t2(agg)
    t3 = make_tables._t3(rows)

    for tex in (t1, t2, t3):
        assert r"\toprule" in tex and r"\bottomrule" in tex and r"\begin{tabular}" in tex

    assert r"\label{tab:main}" in t1
    assert r"\label{tab:kablation}" in t2
    # T2 has one data row per K (between midrule and bottomrule)
    body = t2.split(r"\midrule")[1].split(r"\bottomrule")[0]
    assert body.count(r"\\") == 3
    # T3 lists every run
    assert t3.count(r"\\") >= len(rows)


def test_delta_pm_formatting():
    assert make_tables._pm((0.0736, 0.012)) == r"$0.074 \pm 0.012$"
