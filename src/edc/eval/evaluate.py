"""End-to-end selective-prediction evaluation. [Phase 3]

Trains a reasoner, runs K-restart inference on three **disjoint** folds (fit / calib / test,
invariant 7), extracts geometry features, computes every nonconformity score (the learned
geometry mapper, the ``1 - rho_basin`` fallback, and the raw-energy baselines that geometry must
beat — invariant 8), and returns the metrics dict that ``experiments/run_experiment.py`` writes to
the ledger. The centrepiece is the paired-bootstrap ΔAURC(raw energy − geometry) falsification
test and the Learn-then-Test abstention guarantee.

Plain NumPy over JAX arrays pulled to host (invariant 1): the only JAX is inside the reused
``train`` / ``restarts.solve`` / curvature core.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from edc.conformal import ltt
from edc.conformal import nonconformity as nc
from edc.conformal.selective import ABSTAIN, selective_predict
from edc.eval import metrics
from edc.geometry.features import geometry_features
from edc.inference import restarts
from edc.seeding import numpy_rng, root_key
from edc.train.train_ebm import train

# alpha levels swept for the coverage-validity figure (F3): achieved selective risk must sit
# at or below the nominal target for a valid distribution-free guarantee.
_ALPHA_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
_COVERAGE_GRID = np.linspace(0.02, 1.0, 50)  # F2 risk-coverage sampling grid


def _fold(fns, params, cfg, task, split: str, data_stream: int, key) -> dict:
    """Run inference on one data fold and return its geometry features + baselines + correctness."""
    rng = numpy_rng(cfg.run.seed, 20, data_stream)
    n = cfg.conformal.n_calib if data_stream == 1 else cfg.eval.n_eval
    batch = task.sample(rng, n, split=split)

    k_solve, k_geom = jax.random.split(key)
    traj = restarts.solve(fns, params, jnp.asarray(batch.x), cfg, k_solve)
    feats, names = geometry_features(traj, fns, params, k_geom, grad_tol=cfg.inference.grad_tol)

    pred, _ = restarts.best_of_n_energy(traj)
    correct = np.asarray(task.evaluate(np.asarray(pred), batch.y)).astype(bool)

    terminal = np.asarray(traj.terminal_energy)  # (B, K)
    return {
        "features": np.asarray(feats),
        "names": names,
        "correct": correct,
        "accuracy": float(correct.mean()),
        # Raw-energy baselines (EBT-style scalar-energy confidence) — higher energy = less
        # confident = higher nonconformity score. These are the signals geometry must beat.
        "energy_min": terminal.min(axis=1),
        "energy_mean": terminal.mean(axis=1),
        "energy_std": terminal.std(axis=1),
        "rho": np.asarray(feats)[:, names.index("basin/rho")],
    }


def _resample_curve(scores, correct) -> dict:
    """Risk-coverage curve resampled onto ``_COVERAGE_GRID`` (compact, ledger-friendly, F2)."""
    coverage, risk = metrics.risk_coverage_curve(scores, correct)
    risk_grid = np.interp(_COVERAGE_GRID, coverage, risk)
    return {"coverage": _COVERAGE_GRID.tolist(), "risk": risk_grid.tolist()}


def _feature_diagnostics(features, names, correct, n_bins: int = 20) -> dict:
    """Per-feature correct-vs-incorrect separation (AUROC) + histograms — the F5 mechanism data.

    Compact and fully regenerable: per feature, the single-feature AUROC and 20-bin histograms of
    the correct and incorrect subsets over the feature's observed range.
    """
    features = np.asarray(features, dtype=float)
    correct = np.asarray(correct).astype(bool)
    auroc, hist = {}, {}
    for j, name in enumerate(names):
        col = features[:, j]
        auroc[name] = metrics.single_feature_auroc(col, correct)
        lo, hi = float(col.min()), float(col.max())
        if hi <= lo:
            hi = lo + 1.0                                   # degenerate/constant feature
        edges = np.linspace(lo, hi, n_bins + 1)
        c_counts, _ = np.histogram(col[correct], bins=edges)
        i_counts, _ = np.histogram(col[~correct], bins=edges)
        hist[name] = {"edges": edges.tolist(),
                      "correct_counts": c_counts.tolist(),
                      "incorrect_counts": i_counts.tolist()}
    return {"names": list(names), "auroc": auroc, "hist": hist}


def evaluate(cfg, task, include_ood: bool = True) -> dict:
    """Train, calibrate, and score geometry vs raw energy on disjoint folds. Returns a metrics dict.

    Folds: ``fit`` (train the nonconformity mapper), ``calib`` (choose the LTT threshold), ``test``
    (report AURC, ΔAURC, and the abstention guarantee). Disjoint and independently seeded so the
    conformal guarantees are valid (invariants 7). ΔAURC and the guarantee are read on ``test``;
    the mapper never sees it.

    ``include_ood=False`` skips the OOD fold (and its ``accuracy_ood``/``aurc_ood`` metrics), one
    quarter of the inference — used by the K-sweep, which only needs the ID falsification numbers.
    """
    params, fns, history = train(cfg, task)

    key = root_key(cfg.run.seed)
    k_fit, k_cal, k_test, k_ood = jax.random.split(key, 4)
    fit = _fold(fns, params, cfg, task, "id", 0, k_fit)
    cal = _fold(fns, params, cfg, task, "id", 1, k_cal)
    test = _fold(fns, params, cfg, task, "id", 2, k_test)
    ood = _fold(fns, params, cfg, task, "ood", 3, k_ood) if include_ood else None

    # --- Fit the geometry nonconformity mapper on the FIT fold only (invariant 7) --------------
    mapper = nc.fit_mapper(fit["features"], fit["correct"])

    def scores_for(fold: dict) -> dict:
        """All nonconformity scores on a fold. Low = confident; answer iff score <= threshold."""
        return {
            "geometry": nc.score(mapper, fold["features"]),
            "rho_basin": nc.rho_basin_score(fold["rho"]),
            "energy_min": fold["energy_min"],
            "energy_mean": fold["energy_mean"],
            "energy_std": fold["energy_std"],
        }

    test_scores = scores_for(test)
    cal_scores = scores_for(cal)
    correct = test["correct"]

    # --- AURC per score + the falsification ΔAURC(raw energy - geometry) ------------------------
    aurc = {name: metrics.aurc(s, correct) for name, s in test_scores.items()}
    energy_names = ("energy_min", "energy_mean", "energy_std")
    best_energy = min(energy_names, key=lambda k: aurc[k])  # geometry's hardest baseline

    n_boot = 2000
    d_min, lo_min, hi_min = metrics.paired_bootstrap_delta_aurc(
        test_scores["energy_min"], test_scores["geometry"], correct, n_boot, cfg.run.seed)
    d_best, lo_best, hi_best = metrics.paired_bootstrap_delta_aurc(
        test_scores[best_energy], test_scores["geometry"], correct, n_boot, cfg.run.seed)
    geometry_wins = bool(d_best > 0 and lo_best > 0)

    # --- LTT abstention guarantee at the configured (alpha, delta) ------------------------------
    alpha, delta = cfg.conformal.alpha, cfg.conformal.delta
    out = ltt.calibrate(cal_scores["geometry"], cal["correct"], alpha, delta)
    ltt_block = _ltt_report(out["lambda_hat"], test_scores["geometry"], correct, alpha, delta)

    # --- Coverage-validity sweep (F3: ID) + OOD stress (F6) -------------------------------------
    # The LTT threshold is always calibrated on the ID calib fold; F6 applies that same threshold
    # to the OOD test fold, where exchangeability is broken, to exhibit the guarantee failing.
    ood_geo = scores_for(ood)["geometry"] if include_ood else None
    ood_correct = ood["correct"] if include_ood else None
    validity = {"alpha_grid": [], "target": [], "achieved_risk": [], "coverage": []}
    ood_validity = ({"target": [], "id_risk": [], "ood_risk": [], "id_coverage": [],
                     "ood_coverage": []} if include_ood else None)
    for a in _ALPHA_GRID:
        lam = ltt.calibrate(cal_scores["geometry"], cal["correct"], a, delta)["lambda_hat"]
        r_sel, cov = _selective_risk_coverage(lam, test_scores["geometry"], correct)
        validity["alpha_grid"].append(a)
        validity["target"].append(a)
        validity["achieved_risk"].append(r_sel)
        validity["coverage"].append(cov)
        if include_ood:
            r_ood, cov_ood = _selective_risk_coverage(lam, ood_geo, ood_correct)
            ood_validity["target"].append(a)
            ood_validity["id_risk"].append(r_sel)
            ood_validity["ood_risk"].append(r_ood)
            ood_validity["id_coverage"].append(cov)
            ood_validity["ood_coverage"].append(cov_ood)

    result = {
        "n_fit": int(fit["correct"].shape[0]),
        "n_calib": int(cal["correct"].shape[0]),
        "n_test": int(test["correct"].shape[0]),
        "k_restarts": int(cfg.inference.k_restarts),
        "accuracy_id": test["accuracy"],
        "base_error": float((~correct).mean()),
        "final_train_loss": float(history["epochs"][-1]["loss"]),
        "feature_names": test["names"],
        "aurc": aurc,
        "best_energy_baseline": best_energy,
        "delta_aurc_vs_energy_min": [d_min, lo_min, hi_min],
        "delta_aurc_vs_best_energy": [d_best, lo_best, hi_best],
        "geometry_wins": geometry_wins,
        "risk_coverage": {name: _resample_curve(s, correct) for name, s in test_scores.items()},
        "ltt": ltt_block,
        "coverage_validity": validity,
        "feature_diagnostics": _feature_diagnostics(test["features"], test["names"], correct),
        "ece_geometry": metrics.ece(1.0 - test_scores["geometry"], correct),
    }
    if include_ood:
        result["accuracy_ood"] = ood["accuracy"]
        result["aurc_ood"] = {n: metrics.aurc(s, ood["correct"])
                              for n, s in scores_for(ood).items()}
        # F6: the ID-calibrated threshold at the configured alpha, evaluated on OOD (expect break).
        r_ood, cov_ood = _selective_risk_coverage(out["lambda_hat"], ood_geo, ood_correct)
        result["ood_ltt"] = {
            "alpha": alpha,
            "lambda_hat": out["lambda_hat"],
            "selective_risk": r_ood,
            "coverage": cov_ood,
            "risk_within_budget": bool(cov_ood == 0 or r_ood <= alpha),
        }
        result["ood_validity"] = ood_validity
    return result


def _selective_risk_coverage(lam, scores, correct):
    """``(selective_risk, coverage)`` at threshold ``lam`` on a fold; abstain-everything if None."""
    if lam is None:
        return 0.0, 0.0  # abstain on everything -> no answered errors, zero coverage
    scores = np.asarray(scores, dtype=float)
    answered = scores <= lam
    cov = float(answered.mean())
    if not answered.any():
        return 0.0, 0.0
    r_sel = float((~np.asarray(correct).astype(bool)[answered]).mean())
    return r_sel, cov


def _ltt_report(lam, scores, correct, alpha, delta) -> dict:
    """Selective-prediction metrics at the LTT-chosen threshold on the test fold."""
    r_sel, cov = _selective_risk_coverage(lam, scores, correct)
    answers = selective_predict(np.zeros_like(np.asarray(correct), dtype=int),
                                scores, lam if lam is not None else float("-inf"))
    abstain_rate = float((answers == ABSTAIN).mean())
    return {
        "alpha": alpha,
        "delta": delta,
        "lambda_hat": lam,
        "coverage": cov,
        "abstain_rate": abstain_rate,
        "selective_risk": r_sel,
        "selective_accuracy": 1.0 - r_sel if cov > 0 else None,
        "risk_within_budget": bool(cov == 0 or r_sel <= alpha),
    }
