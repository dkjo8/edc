# Decisions (ADR log)

Short, dated, append-only. Newest first.

## 2026-07-24 — JAX as the numeric core
**Decision:** JAX/Flax for `energy/`, `inference/`, `geometry/curvature.py`; everything else
plain Python behind `edc.energy.base.ReasonerFns`.
**Why:** the crux workload is `vmap` over K restarts + per-input Hessian-vector products
(`jvp∘grad`) + a `lax.scan` inner loop + gradient-through-optimization. JAX composes these
natively with far less friction than `torch.func`, and its deterministic `PRNGKey`/`fold_in`
makes "regenerate figure from seed" exact. No PyTorch EBRM checkpoint exists to reuse (EBRM is
Julia), so nothing is lost by not matching Tars's PyTorch stack.
**Reversible?** Yes — JAX is isolated to three module areas behind one interface.

## 2026-07-24 — Selective vs halting use different conformal tools
**Decision:** Conformal Risk Control for halting; Learn-then-Test for selective/abstention.
**Why:** halting risk is **monotone** in one threshold (CRC's assumption); selective risk
`P(err|answered)` is **non-monotone**, which CRC cannot handle but LTT can (hypothesis testing
with valid p-values). Using the wrong tool would silently void the guarantee.

## 2026-07-24 — Phase-1 training uses a fixed per-class codebook anchor
**Decision:** shape the landscape with contrastive + decode losses against a frozen random
per-class latent anchor, rather than the full IRED annealed-landscape + score-matching recipe.
**Why:** minimal, fast (<60s CPU smoke), and genuinely produces basins to analyse in Phase 2.
**Follow-up:** upgrade to IRED-style training in Phase 4 (tracked in `RESEARCH_PLAN.md`).

## 2026-07-24 — Sticky append-only JSONL ledger is the single source of truth
**Decision:** all results land in `results/ledger.jsonl` via `edc.ledger`; figures/tables read
only from it. **Why:** matches Tars conventions; makes every paper number traceable to a
`(git_sha, config_hash, seed)`.
