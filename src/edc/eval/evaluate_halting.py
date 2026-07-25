"""End-to-end adaptive-halting evaluation (the CRC guarantee). [Phase 4b]

Trains a reasoner, records the per-step decoded answers on disjoint calib/test folds
(``restarts.solve(record_steps=True)``), CRC-calibrates the basin-agreement halting threshold on
calib, and reports on test: compute saved, halting risk (disagreement with the full-budget answer)
vs the budget ``alpha``, end-task accuracy retained, and a tau-sweep for the F4 Pareto. Returns the
JSON-safe metrics dict that ``experiments/run_halting.py`` writes to the ledger.

Plain NumPy over JAX arrays pulled to host (invariant 1); the only JAX is the reused train/solve
core. Randomness via ``edc.seeding`` (invariant 4).
"""

from __future__ import annotations

import jax
import numpy as np

from edc.halting import adaptive
from edc.inference import restarts
from edc.seeding import numpy_rng, root_key
from edc.train.train_ebm import train


def _fold(fns, params, cfg, task, data_stream: int, key) -> dict:
    """Sample a fold and run K-restart inference with per-step decoding recorded."""
    rng = numpy_rng(cfg.run.seed, 30, data_stream)
    n = cfg.conformal.n_calib if data_stream == 0 else cfg.eval.n_eval
    batch = task.sample(rng, n, split="id")
    import jax.numpy as jnp

    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, key, record_steps=True)
    pred_full, _ = restarts.best_of_n_energy(traj)
    return {
        "step_pred": np.asarray(traj.step_pred),      # (B, K, T+1)
        "energies": np.asarray(traj.energies),        # (B, K, T+1)
        "y": np.asarray(batch.y),
        "pred_full": np.asarray(pred_full),           # best-of-N at full budget
    }


def _accuracy(pred, y) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(y)))


def evaluate_halting(cfg, task) -> dict:
    """Train, CRC-calibrate the halting threshold on calib, and report the test compute/accuracy."""
    params, fns, history = train(cfg, task)
    key = root_key(cfg.run.seed)
    k_cal, k_test = jax.random.split(key)
    cal = _fold(fns, params, cfg, task, 0, k_cal)
    test = _fold(fns, params, cfg, task, 1, k_test)

    K = int(cfg.inference.k_restarts)
    # Candidate agreement thresholds: multiples of 1/K span the achievable agreement range.
    taus = np.round(np.linspace(1.0 / K, 1.0, K), 4)

    cal_out = adaptive.calibrate(cal["step_pred"], cal["energies"], cfg.conformal.alpha, taus)
    tau_hat = cal_out["tau_hat"]

    full_acc = _accuracy(test["pred_full"], test["y"])

    # tau-sweep on the TEST fold for F4: compute used, accuracy vs labels, disagreement with full.
    sweep = {"taus": taus.tolist(), "compute_used": [], "accuracy": [], "disagreement": []}
    for tau in taus:
        early, compute = adaptive.halted_predictions(test["step_pred"], test["energies"], tau)
        sweep["compute_used"].append(float(compute.mean()))
        sweep["accuracy"].append(_accuracy(early, test["y"]))
        sweep["disagreement"].append(float(np.mean(early != test["pred_full"])))

    # The CRC-chosen operating point on test.
    if tau_hat is not None:
        early, compute = adaptive.halted_predictions(test["step_pred"], test["energies"], tau_hat)
        halting_risk = float(np.mean(early != test["pred_full"]))
        halted_acc = _accuracy(early, test["y"])
        compute_used = float(compute.mean())
    else:
        halting_risk, halted_acc, compute_used = 0.0, full_acc, 1.0   # infeasible => never halt

    return {
        "k_restarts": K,
        "steps": int(cfg.inference.steps),
        "alpha": float(cfg.conformal.alpha),
        "n_calib": int(cal["y"].shape[0]),
        "n_test": int(test["y"].shape[0]),
        "final_train_loss": float(history["epochs"][-1]["loss"]),
        "tau_hat": tau_hat,
        "compute_used": compute_used,
        "compute_saved": 1.0 - compute_used,
        "halting_risk": halting_risk,
        "risk_within_budget": bool(halting_risk <= cfg.conformal.alpha),
        "full_accuracy": full_acc,
        "halted_accuracy": halted_acc,
        "accuracy_drop": full_acc - halted_acc,
        "risk_monotone_in_tau": cal_out["risk_monotone_in_tau"],
        "calib_risk_by_tau": cal_out["risk_by_tau"],
        "calib_compute_by_tau": cal_out["compute_by_tau"],
        "tau_sweep": sweep,
    }
