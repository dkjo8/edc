"""Conformal Risk Control for adaptive halting. [Phase 3]

The halting risk ``L(lambda) = 1[early-stopped answer != full-budget answer]`` is bounded and
**monotone** in a single threshold ``lambda`` (relax lambda => stop earlier => weakly more
errors). CRC (Angelopoulos et al., arXiv:2208.02814) then picks ``lambda_hat`` so
``E[L(lambda_hat)] <= alpha`` in finite samples. Monotonicity is exactly why CRC — not LTT —
is the right tool here.
"""

from __future__ import annotations


def calibrate(loss_by_lambda, alpha: float):
    raise NotImplementedError("Phase 3: choose the smallest-compute lambda with E[L] <= alpha.")
