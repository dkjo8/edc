"""Adaptive halting: exact per-step agreement, stop-step policy, halting losses, and CRC
calibration (with the lambda = 1 - tau orientation). Offline + CPU-only, hand-built trajectories.
"""

import numpy as np

from edc.halting import adaptive as H


def test_per_step_agreement_exact():
    # 1 input, K=4 restarts, T+1=3 steps. t0: {0,1,2,3} all differ -> 1/4; t1: {5,5,5,6} -> 3/4;
    # t2: all 5 -> 1.0.
    step_pred = np.array([[[0, 5, 5], [1, 5, 5], [2, 5, 5], [3, 6, 5]]])   # (1,4,3)
    agr = H.per_step_agreement(step_pred)
    assert np.allclose(agr[0], [0.25, 0.75, 1.0])


def test_halting_policy_first_crossing_else_full():
    a = np.array([0.25, 0.5, 0.75, 1.0])
    assert H.halting_policy(a, 0.75) == 2          # first step reaching 0.75
    assert H.halting_policy(a, 1.0) == 3           # only the last step is unanimous
    assert H.halting_policy(a, 1.5) == 3           # never reached -> full budget (last index)


def test_halting_losses_early_vs_full():
    # K=2, T+1=3. Restart 0 always lowest energy => best-of-N follows restart 0.
    # Restart-0 answers: step0=9, step1=9, step2=7 (full). Agreement: t0 {9,1}->0.5, t1 {9,2}->0.5,
    # t2 {7,7}->1.0. At tau=1.0 we stop at t2 => early == full (7) => loss 0. At tau=0.5 we stop at
    # t0 => early=9 != full=7 => loss 1, compute 0.
    step_pred = np.array([[[9, 9, 7], [1, 2, 7]]])                          # (1,2,3)
    energies = np.array([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]])               # restart 0 lower
    loss, compute = H.halting_losses(step_pred, energies, taus=[1.0, 0.5])
    assert loss[0, 0] == 0.0 and np.isclose(compute[0, 0], 1.0)             # tau=1.0: full budget
    assert loss[0, 1] == 1.0 and np.isclose(compute[0, 1], 0.0)             # tau=0.5: halt at t0


def test_halted_predictions_matches_policy():
    step_pred = np.array([[[9, 9, 7], [1, 2, 7]]])
    energies = np.array([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]])
    early, compute = H.halted_predictions(step_pred, energies, tau=1.0)
    assert early[0] == 7 and np.isclose(compute[0], 1.0)


def _synthetic_cal(n=400, seed=0):
    """Build (step_pred, energies) where higher tau (later stop) strictly lowers disagreement:
    each input agrees early with prob p; a late-stop always matches the full answer."""
    rng = np.random.default_rng(seed)
    K, T1 = 3, 5
    step_pred = np.zeros((n, K, T1), dtype=int)
    energies = np.zeros((n, K, T1))
    energies[:, 1:, :] = 1.0                       # restart 0 always lowest -> best-of-N follows it
    full = rng.integers(0, 5, size=n)
    for b in range(n):
        step_pred[b, 0, -1] = full[b]             # full answer at last step
        early_ok = rng.random() < 0.7
        step_pred[b, 0, :-1] = full[b] if early_ok else (full[b] + 1) % 5   # early maybe wrong
        step_pred[b, 1:, :] = (full[b] + 2) % 5  # other restarts disagree -> low early agreement
        step_pred[b, :, -1] = full[b]            # all agree at the end -> agreement 1.0 at end
    return step_pred, energies


def test_crc_calibration_controls_halting_risk():
    step_pred, energies = _synthetic_cal(n=600, seed=1)
    alpha, K = 0.1, 3
    taus = np.round(np.linspace(1.0 / K, 1.0, K), 4)
    out = H.calibrate(step_pred, energies, alpha, taus)
    # A feasible budget picks a tau whose empirical risk is within alpha; risk falls as tau rises.
    assert out["tau_hat"] is not None
    risk = np.array(out["risk_by_tau"])
    assert np.all(np.diff(risk) <= 1e-9)                     # non-increasing in tau (monotone)
    chosen_idx = list(taus).index(out["tau_hat"])
    assert risk[chosen_idx] <= alpha                         # chosen point respects the budget


def test_calibrate_infeasible_budget_returns_none():
    step_pred, energies = _synthetic_cal(n=200, seed=2)
    out = H.calibrate(step_pred, energies, alpha=0.0, taus=[0.34, 0.67, 1.0])
    # alpha=0 with finite calibration is structurally infeasible (CRC (n+1) correction) -> None.
    assert out["tau_hat"] is None
