import numpy as np

from edc.seeding import numpy_rng, restart_keys, root_key


def test_root_key_deterministic():
    assert np.array_equal(np.asarray(root_key(0)), np.asarray(root_key(0)))
    assert not np.array_equal(np.asarray(root_key(0)), np.asarray(root_key(1)))


def test_restart_keys_are_prefix_stable():
    # fold_in indexing => restart k's key is independent of how many restarts we ask for.
    k8 = np.asarray(restart_keys(0, 8))
    k4 = np.asarray(restart_keys(0, 4))
    assert np.array_equal(k8[:4], k4)
    # distinct restarts get distinct keys
    assert not np.array_equal(k8[0], k8[1])


def test_numpy_streams_are_independent_and_reproducible():
    a1 = numpy_rng(0, 1).standard_normal(5)
    a2 = numpy_rng(0, 1).standard_normal(5)
    b = numpy_rng(0, 2).standard_normal(5)
    assert np.array_equal(a1, a2)          # reproducible
    assert not np.array_equal(a1, b)       # different substream
