"""Learn-then-Test for selective prediction / abstention. [Phase 3]

Selective risk ``R_sel(lambda) = P(error | answered)`` is non-monotone in the threshold, so
plain conformal/CRC does not apply. LTT (Angelopoulos et al., arXiv:2110.01052) treats each
candidate ``lambda`` as a hypothesis ``H_lambda: R_sel(lambda) > alpha``, computes a valid
Hoeffding-Bentkus p-value, and returns the FWER-controlled admissible set. Guarantee:
``P(R_sel(lambda_hat) <= alpha) >= 1 - delta``, distribution-free, under exchangeability.
"""

from __future__ import annotations


def hoeffding_bentkus_pvalue(risk_hat: float, alpha: float, n: int) -> float:
    raise NotImplementedError("Phase 3: valid p-value for H: R_sel > alpha.")


def calibrate(scores_cal, correct_cal, alpha: float, delta: float):
    raise NotImplementedError("Phase 3: return admissible thresholds controlling selective risk.")
