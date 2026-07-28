"""Phase-4k full-Hessian-spectrum features: exact eigenvalues on a known quadratic energy, and
the derived spectrum/* summaries against closed-form values.

Offline + CPU-only (conftest forces JAX_PLATFORMS=cpu). The Hessian of ``E(z)=½ zᵀAz`` is the
constant matrix ``A`` for every ``z``, so ``eigvalsh`` must recover ``A``'s spectrum exactly.
"""

import types

import jax.numpy as jnp
import numpy as np

from edc.geometry.curvature import batched_spectrum
from edc.geometry.spectrum import spectrum_features
from edc.inference.trajectory import TrajectoryRecord

_SPECTRUM_NAMES = [
    "spectrum/lmin_mean", "spectrum/lmin_best",
    "spectrum/negfrac_mean", "spectrum/negfrac_best",
    "spectrum/effrank_mean", "spectrum/effrank_best",
    "spectrum/logdet_mean", "spectrum/logdet_best",
]


def _spd(d, seed):
    A = np.random.default_rng(seed).standard_normal((d, d)).astype(np.float64)
    return A @ A.T + np.eye(d)                                # symmetric positive definite


def _quadratic_energy(A):
    Aj = jnp.asarray(A)

    def energy(_params, _h, z):                              # (n, d) -> (n,)
        return 0.5 * jnp.einsum("ni,ij,nj->n", z, Aj, z)

    return energy


def test_batched_spectrum_matches_eigvalsh():
    d, n = 5, 4
    A = _spd(d, 0)
    zs = jnp.asarray(np.random.default_rng(1).standard_normal((n, d)))
    contexts = jnp.zeros((n, 1))
    evals = np.asarray(batched_spectrum(_quadratic_energy(A), {}, contexts, zs))
    assert evals.shape == (n, d)
    ref = np.linalg.eigvalsh(A)                              # ascending
    for row in evals:                                        # Hessian is constant A for every z
        assert np.allclose(row, ref, atol=1e-6)


def test_spectrum_features_shape_names_and_values():
    d, B, K = 4, 2, 3
    A = _spd(d, 2)
    z = np.random.default_rng(3).standard_normal((B, K, d))
    energies = np.zeros((B, K, 2))                           # terminal energy 0 -> best = restart 0
    traj = TrajectoryRecord(
        z_star=z, energies=energies, grad_norms=np.zeros((B, K, 1)),
        logits=np.zeros((B, K, 1)), pred=np.zeros((B, K), int), h_x=np.zeros((B, 1)),
    )
    fns = types.SimpleNamespace(energy=_quadratic_energy(A))
    feats, names = spectrum_features(traj, fns, {})
    assert names == _SPECTRUM_NAMES
    assert feats.shape == (B, 8)
    assert np.all(np.isfinite(feats))

    eig = np.linalg.eigvalsh(A)                              # same for every particle (constant H)
    lmin = eig[0]
    effrank = eig.sum() ** 2 / (eig**2).sum()
    logdet = np.sum(np.log(eig + 1e-8))
    # mean-over-K == best (spectrum identical across restarts); SPD => no negative eigenvalues.
    for row in feats:
        assert np.allclose(row[0:2], lmin, atol=1e-6)        # lmin_mean, lmin_best
        assert np.allclose(row[2:4], 0.0)                    # negfrac_mean, negfrac_best
        assert np.allclose(row[4:6], effrank, atol=1e-5)     # effrank
        assert np.allclose(row[6:8], logdet, atol=1e-5)      # logdet


def test_spectrum_detects_negative_eigenvalue_at_saddle():
    # Indefinite Hessian diag(1, -1): one negative eigenvalue => negfrac = 0.5, lmin < 0.
    A = np.diag([1.0, -1.0])
    z = np.zeros((1, 1, 2))
    traj = TrajectoryRecord(
        z_star=z, energies=np.zeros((1, 1, 2)), grad_norms=np.zeros((1, 1, 1)),
        logits=np.zeros((1, 1, 1)), pred=np.zeros((1, 1), int), h_x=np.zeros((1, 1)),
    )
    fns = types.SimpleNamespace(energy=_quadratic_energy(A))
    feats, names = spectrum_features(traj, fns, {})
    lmin_mean = feats[0, names.index("spectrum/lmin_mean")]
    negfrac_mean = feats[0, names.index("spectrum/negfrac_mean")]
    assert lmin_mean < 0.0
    assert np.isclose(negfrac_mean, 0.5)
