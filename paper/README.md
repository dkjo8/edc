# paper/

LaTeX source for the EDC paper. Build:

```bash
cd paper && latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main x2
```

## Figure/table → claim map

Every figure/table is regenerated from `results/ledger.jsonl` by `analysis/make_figures.py` /
`make_tables.py` — never hand-drawn (except F1, the schematic). Each analysis function records
the ledger `run_id`s it consumed.

| Artifact | Claim it supports | Source |
|----------|-------------------|--------|
| F1 | The method: encode → K-restart descent → geometry → certificate | TikZ (hand) |
| F2 | Geometry beats scalar-energy/MSP/MC-dropout on risk–coverage | `make_figures` |
| F3 | The conformal guarantee holds in-distribution (coverage on diagonal) | `make_figures` |
| F4 | Adaptive halting matches full-budget accuracy at less compute | `make_figures` |
| F5 | Basin/curvature separate correct from incorrect (the mechanism) | `make_figures` |
| F6 | Guarantee degrades under OOD → motivates abstention | `make_figures` |
| T1 | Main results across tasks | `make_tables` |
| T2 | Ablations: feature groups, K, τ, curvature fidelity | `make_tables` |
| T3 | Reproducibility appendix (seeds/configs/env) | `make_tables` |

## Target

NeurIPS/ICML main track (methods) or UAI/AISTATS (conformal angle); arXiv `cs.LG` + `stat.ML`.
Prepare artifact-evaluation from day one — the whole repo is built so `make figures && make
tables && latexmk` rebuilds the PDF from seeds + configs.
