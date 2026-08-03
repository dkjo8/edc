import numpy as np

from edc.seeding import member_seeds, numpy_rng, restart_keys, root_key


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


def test_member_seeds_deterministic_distinct_and_disjoint():
    s0 = member_seeds(0, 4)
    assert len(s0) == 4 and len(set(s0.tolist())) == 4          # deterministic count, all distinct
    assert np.array_equal(s0, member_seeds(0, 4))               # pure function of the base seed
    # different base seeds give disjoint member sets (no cross-seed ensemble leakage)
    assert not set(s0.tolist()) & set(member_seeds(1, 4).tolist())
    assert member_seeds(0, 0).shape == (0,)                     # M-1 == 0 -> empty (ensemble off)
