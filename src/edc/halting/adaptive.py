"""Geometry-driven adaptive halting policy. [Phase 4b]

Stop the K-restart descent as soon as **basin agreement** — the plurality fraction of decoded
answers across restarts — crosses a threshold ``tau``, saving compute versus the fixed step budget.
The halting loss ``L(tau) = 1[early-stopped best-of-N answer != full-budget best-of-N answer]`` is
bounded in [0,1]; Conformal Risk Control (``edc.conformal.crc``) picks the most compute-saving
``tau`` with ``E[L] <= alpha`` in finite samples. Produces the compute-vs-accuracy Pareto (F4).

Plain NumPy over ``TrajectoryRecord.step_pred`` / ``energies`` (invariant 1); the reparametrisation
``lambda = 1 - tau`` orients the sweep so CRC's "largest admissible lambda = most compute saved"
convention selects the most aggressive safe halt.
"""

from __future__ import annotations

import numpy as np

from edc.conformal import crc


def per_step_agreement(step_pred) -> np.ndarray:
    """``(B, T+1)`` basin agreement at each descent step: plurality fraction across the K restarts.

    ``step_pred`` is ``(B, K, T+1)`` decoded classes (from ``restarts.solve(record_steps=True)``).
    1.0 => all restarts decode the same answer at that step; low => scattered. Same plurality
    definition as ``geometry.basin.basin_features``, vectorised over the step axis.
    """
    step_pred = np.asarray(step_pred)
    B, K, T1 = step_pred.shape
    n_classes = int(step_pred.max()) + 1
    counts = np.zeros((B, T1, n_classes), dtype=float)
    b_idx = np.arange(B)[:, None, None]
    t_idx = np.arange(T1)[None, None, :]
    np.add.at(counts, (b_idx, t_idx, step_pred), 1.0)         # (B, T+1, C)
    return counts.max(axis=2) / K                            # (B, T+1)


def halting_policy(agreement_row, tau: float) -> int:
    """First step index whose agreement ``>= tau`` for one input; else the full budget ``T``.

    ``agreement_row`` is ``(T+1,)``. Returns an index in ``[0, T]`` (T = last = full budget).
    """
    a = np.asarray(agreement_row, dtype=float)
    hits = np.nonzero(a >= tau)[0]
    return int(hits[0]) if hits.size else a.shape[0] - 1


def _stop_steps(agreement: np.ndarray, tau: float) -> np.ndarray:
    """Vectorised ``halting_policy`` over inputs: ``(B,)`` stop indices at threshold ``tau``."""
    mask = agreement >= tau                                   # (B, T+1)
    T = agreement.shape[1] - 1
    return np.where(mask.any(axis=1), mask.argmax(axis=1), T)


def _best_of_n_at(step_pred: np.ndarray, energies: np.ndarray, stop_t: np.ndarray) -> np.ndarray:
    """Best-of-N (lowest-energy restart) decoded answer at per-input step ``stop_t`` -> ``(B,)``."""
    B, K, _ = step_pred.shape
    rows = np.arange(B)
    e_at = energies[rows[:, None], np.arange(K)[None, :], stop_t[:, None]]   # (B, K)
    k_best = e_at.argmin(axis=1)                                             # (B,)
    return step_pred[rows, k_best, stop_t]                                   # (B,)


def halted_predictions(step_pred, energies, tau: float):
    """``(early_pred (B,), compute_used (B,))`` under the halting policy at threshold ``tau``.

    The reported answer is best-of-N (lowest energy) at the agreement-triggered stop step; compute
    used is the fraction of the step budget consumed. Used by the evaluator (accuracy vs labels) and
    the F4 Pareto.
    """
    step_pred = np.asarray(step_pred)
    energies = np.asarray(energies, dtype=float)
    T = step_pred.shape[2] - 1
    stop_t = _stop_steps(per_step_agreement(step_pred), tau)
    return _best_of_n_at(step_pred, energies, stop_t), stop_t / T


def halting_losses(step_pred, energies, taus):
    """Per-input halting loss and compute fraction for each ``tau``.

    Returns ``(loss, compute_used)`` each ``(B, n_tau)``: ``loss = 1[early answer != full answer]``
    where the early answer is best-of-N at the agreement-triggered stop step and the full answer is
    best-of-N at the final step; ``compute_used = stop_step / T`` (fraction of the step budget).
    """
    step_pred = np.asarray(step_pred)
    energies = np.asarray(energies, dtype=float)
    B, K, T1 = step_pred.shape
    T = T1 - 1
    agreement = per_step_agreement(step_pred)
    full = _best_of_n_at(step_pred, energies, np.full(B, T))     # best-of-N at the final step

    taus = np.asarray(taus, dtype=float)
    loss = np.empty((B, taus.shape[0]))
    compute = np.empty((B, taus.shape[0]))
    for j, tau in enumerate(taus):
        stop_t = _stop_steps(agreement, tau)
        early = _best_of_n_at(step_pred, energies, stop_t)
        loss[:, j] = (early != full).astype(float)
        compute[:, j] = stop_t / T
    return loss, compute


def calibrate(step_pred_cal, energies_cal, alpha: float, taus) -> dict:
    """CRC-calibrate the halting threshold ``tau`` at error budget ``alpha`` on a calibration fold.

    Reparametrises ``lambda = 1 - tau`` so ascending ``lambda`` means a lower agreement bar => stop
    earlier => weakly more disagreement and more compute saved, matching ``crc.calibrate``'s
    "largest admissible lambda = most compute saved". Returns ``tau_hat`` (most aggressive safe
    threshold, or ``None`` if the budget is infeasible) plus the per-tau risk/compute curves and a
    monotonicity flag (CRC assumes the risk is monotone in the threshold).
    """
    taus = np.asarray(taus, dtype=float)
    loss, compute = halting_losses(step_pred_cal, energies_cal, taus)
    risk_by_tau = loss.mean(axis=0)
    compute_by_tau = compute.mean(axis=0)

    lambda_hat = crc.calibrate(loss, alpha, lambdas=1.0 - taus)
    tau_hat = None if lambda_hat is None else float(1.0 - lambda_hat)

    # Risk should be non-increasing as tau rises (later stop). Flag material violations.
    monotone = bool(np.all(np.diff(risk_by_tau) <= 1e-9))
    return {
        "tau_hat": tau_hat,
        "alpha": float(alpha),
        "taus": taus.tolist(),
        "risk_by_tau": risk_by_tau.tolist(),
        "compute_by_tau": compute_by_tau.tolist(),
        "risk_monotone_in_tau": monotone,
    }
