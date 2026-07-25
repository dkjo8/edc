"""Learn-then-Test for selective prediction / abstention. [Phase 3]

Selective risk ``R_sel(lambda) = P(error | answered)`` is non-monotone in the threshold, so
plain conformal/CRC does not apply. LTT (Angelopoulos et al., arXiv:2110.01052) treats each
candidate ``lambda`` as a hypothesis ``H_lambda: R_sel(lambda) > alpha``, computes a valid
Hoeffding-Bentkus p-value, and returns the FWER-controlled admissible set. Guarantee:
``P(R_sel(lambda_hat) <= alpha) >= 1 - delta``, distribution-free, under exchangeability.

NumPy / scipy only (invariant 1).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import rel_entr
from scipy.stats import binom


def hoeffding_bentkus_pvalue(risk_hat: float, alpha: float, n: int) -> float:
    """Valid p-value for ``H: R > alpha`` given empirical mean loss ``risk_hat`` over ``n`` points.

    ``p = min(p_Hoeffding, p_Bentkus, 1)``. Rejecting ``H`` at level ``delta`` certifies
    ``R <= alpha``. The loss is in ``[0, 1]`` (here a selective error rate).

    - Hoeffding (relative-entropy / Chernoff form, tighter than the additive bound):
      ``exp(-n * KL(risk_hat || alpha))`` with ``KL`` via ``rel_entr`` so ``0*log 0 -> 0``.
    - Bentkus: ``e * P(Bin(n, alpha) <= ceil(n * risk_hat))``.

    Returns ``1.0`` when ``risk_hat >= alpha`` (no evidence to reject) or ``n <= 0`` (an empty
    sample certifies nothing) — the Hoeffding form is only valid for ``risk_hat < alpha``.
    """
    if n <= 0 or risk_hat >= alpha:
        return 1.0
    risk_hat = max(float(risk_hat), 0.0)
    kl = float(rel_entr(risk_hat, alpha) + rel_entr(1.0 - risk_hat, 1.0 - alpha))
    p_hoeffding = math.exp(-n * kl)
    p_bentkus = math.e * float(binom.cdf(math.ceil(n * risk_hat), n, alpha))
    return float(min(p_hoeffding, p_bentkus, 1.0))


def selective_risk(scores_cal, correct_cal, lam: float) -> tuple[float, int]:
    """``(R_sel, n_answered)`` at threshold ``lam``: error rate among answered inputs (s <= lam)."""
    scores = np.asarray(scores_cal, dtype=float)
    correct = np.asarray(correct_cal).astype(bool)
    answered = scores <= lam
    n_ans = int(answered.sum())
    if n_ans == 0:
        return float("nan"), 0
    return float((~correct[answered]).mean()), n_ans


def calibrate(scores_cal, correct_cal, alpha: float, delta: float, lambdas=None,
              n_grid: int = 100) -> dict:
    """Return the FWER-controlled admissible thresholds and the max-coverage pick.

    For each candidate threshold ``lambda`` the selective risk ``R_sel`` is the error rate among
    answered inputs (``s <= lambda``) over ``n_ans`` points, with p-value ``HB(R_sel, alpha,
    n_ans)`` for ``H_lambda: R_sel > alpha``. Because ``R_sel`` is **non-monotone** in ``lambda``,
    fixed-sequence testing does not apply (the smallest thresholds answer too few points to have
    any statistical power, so a sequence would stall immediately); we instead use **Bonferroni**
    over a bounded quantile grid of ``m`` candidates: admissible ``= {lambda : p(lambda) <=
    delta/m}``, which controls FWER at ``delta`` for arbitrary thresholds. ``lambda_hat`` is the
    largest admissible threshold (max coverage), or ``None`` if none pass — abstain on everything
    at this ``(alpha, delta)`` given ``n``.

    An empty answered set (``lambda`` below every score) gives ``p = 1`` (never admissible; no 0/0).
    Pass explicit ``lambdas`` to override the grid.
    """
    scores = np.asarray(scores_cal, dtype=float)
    correct = np.asarray(correct_cal).astype(bool)
    if lambdas is None:
        # Quantile grid keeps the multiplicity m bounded (so delta/m is not crushed) while still
        # placing candidates where the scores actually are.
        lambdas = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_grid)))
    lambdas = np.sort(np.asarray(lambdas, dtype=float))
    m = lambdas.shape[0]

    pvalues = np.empty(m, dtype=float)
    for i, lam in enumerate(lambdas):
        r_sel, n_ans = selective_risk(scores, correct, lam)
        pvalues[i] = 1.0 if n_ans == 0 else hoeffding_bentkus_pvalue(r_sel, alpha, n_ans)

    admissible = lambdas[pvalues <= delta / m]  # Bonferroni
    lambda_hat = float(admissible.max()) if admissible.size else None
    return {"admissible": admissible, "lambda_hat": lambda_hat, "pvalues": pvalues,
            "lambdas": lambdas, "m": m}
