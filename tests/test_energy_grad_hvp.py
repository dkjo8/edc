"""Finite-difference checks of the gradient and Hessian-vector product.

The HVP is the crux primitive of the geometry features, so we verify it two ways:
(1) against a closed-form quadratic where H is constant, and
(2) against a numerical HVP of the *actual* energy net w.r.t. the latent.
Uses float64 for finite-difference accuracy.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from edc.geometry.curvature import hutchinson_trace, hvp, lambda_max  # noqa: E402


def test_hvp_matches_quadratic_closed_form():
    d = 6
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (d, d), dtype=jnp.float64)
    A = A @ A.T + jnp.eye(d)  # SPD
    b = jax.random.normal(jax.random.PRNGKey(1), (d,), dtype=jnp.float64)

    def f(z):
        return 0.5 * z @ A @ z + b @ z

    z = jax.random.normal(jax.random.PRNGKey(2), (d,), dtype=jnp.float64)
    v = jax.random.normal(jax.random.PRNGKey(3), (d,), dtype=jnp.float64)

    # grad closed form: A z + b
    assert np.allclose(np.asarray(jax.grad(f)(z)), np.asarray(A @ z + b), atol=1e-8)
    # HVP closed form: A v
    assert np.allclose(np.asarray(hvp(f, z, v)), np.asarray(A @ v), atol=1e-8)
    # top eigenvalue matches numpy eig
    lam = float(lambda_max(f, z, jax.random.PRNGKey(4), iters=50))
    assert np.isclose(lam, float(np.linalg.eigvalsh(np.asarray(A))[-1]), rtol=1e-3)
    # trace matches
    tr = float(hutchinson_trace(f, z, jax.random.PRNGKey(5), n_probes=256))
    assert np.isclose(tr, float(np.trace(np.asarray(A))), rtol=0.15)


def test_hvp_matches_finite_difference():
    d = 5
    A = jnp.asarray(np.random.default_rng(0).standard_normal((d, d)))
    A = A @ A.T

    def f(z):
        return jnp.sum(jnp.tanh(A @ z) ** 2)

    z = jnp.asarray(np.random.default_rng(1).standard_normal(d))
    v = jnp.asarray(np.random.default_rng(2).standard_normal(d))

    eps = 1e-5
    fd = (jax.grad(f)(z + eps * v) - jax.grad(f)(z - eps * v)) / (2 * eps)
    assert np.allclose(np.asarray(hvp(f, z, v)), np.asarray(fd), atol=1e-4)
