"""Phase-4k mode-connectivity features: energy barriers along the path between restart endpoints.

Offline + CPU-only. A convex bowl has no barrier between any two points; a double well
``E(z)=(z²-1)²`` has a barrier of height 1 between its two minima at ``±1``.
"""

import types

import jax.numpy as jnp
import numpy as np

from edc.geometry.connectivity import connectivity_features
from edc.inference.trajectory import TrajectoryRecord

_CONNECT_NAMES = ["connect/barrier_mean", "connect/barrier_max", "connect/connected_frac"]


def _traj(z_star, terminal_energy):
    z = np.asarray(z_star, dtype=float)
    B, K, _ = z.shape
    te = np.asarray(terminal_energy, dtype=float)            # (B, K)
    energies = te[:, :, None]                                # (B, K, 1): terminal_energy = [...,-1]
    return TrajectoryRecord(
        z_star=z, energies=energies, grad_norms=np.zeros((B, K, 1)),
        logits=np.zeros((B, K, 1)), pred=np.zeros((B, K), int), h_x=np.zeros((B, 1)),
    )


def _bowl():
    def energy(_params, _h, z):                              # convex: E = ½|z|²
        return 0.5 * jnp.sum(z**2, axis=-1)
    return types.SimpleNamespace(energy=energy)


def _double_well():
    def energy(_params, _h, z):                              # (z²-1)² 1-D: wells at ±1, hump 1 at 0
        return (z[:, 0] ** 2 - 1.0) ** 2
    return types.SimpleNamespace(energy=energy)


def test_names_and_k1_returns_zeros():
    traj = _traj(np.zeros((2, 1, 1)), np.zeros((2, 1)))
    feats, names = connectivity_features(traj, _bowl(), {})
    assert names == _CONNECT_NAMES
    assert feats.shape == (2, 3)
    assert np.allclose(feats, 0.0)                           # K<2 -> no pairs


def test_identical_endpoints_zero_barrier():
    z = np.full((1, 2, 1), 0.5)                              # both restarts at the same point
    feats, _ = connectivity_features(_traj(z, np.zeros((1, 2))), _bowl(), {})
    assert np.allclose(feats[0, :2], 0.0)                    # barrier_mean, barrier_max
    assert np.isclose(feats[0, 2], 1.0)                      # connected_frac: same basin


def test_convex_landscape_no_barrier():
    z = np.array([[[0.0], [2.0]]])                           # two distinct points, convex bowl
    feats, _ = connectivity_features(_traj(z, np.array([[0.0, 2.0]])), _bowl(), {})
    assert np.allclose(feats[0, :2], 0.0)                    # a line in a convex bowl never climbs
    assert np.isclose(feats[0, 2], 1.0)


def test_double_well_positive_barrier():
    z = np.array([[[-1.0], [1.0]]])                          # the two wells
    feats, names = connectivity_features(_traj(z, np.array([[0.0, 0.0]])), _double_well(), {})
    barrier_mean = feats[0, names.index("connect/barrier_mean")]
    connected = feats[0, names.index("connect/connected_frac")]
    # hump at z=0 has E=1, endpoints E=0; the 8-point grid samples just off the peak (~0.96 <= 1).
    assert 0.9 < barrier_mean <= 1.0 + 1e-6
    assert np.isclose(connected, 0.0)                        # separated basins
