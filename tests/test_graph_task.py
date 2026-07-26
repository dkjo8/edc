"""GraphPlanningTask: BFS shortest-path labels, fixed-size padded encoding, size-shift OOD, and
determinism. Offline + CPU-only, pure NumPy (no training).
"""

import numpy as np

from edc.registry import build_task
from edc.tasks.graph_planning import _MAX_NODES, GraphPlanningTask, _shortest_len


def _chain(n):
    adj = np.zeros((n, n))
    for a in range(n - 1):
        adj[a, a + 1] = adj[a + 1, a] = 1.0
    return adj


def test_bfs_shortest_len():
    adj = _chain(5)                                   # 0-1-2-3-4
    assert _shortest_len(adj, 0, 4, cap=5) == 4
    assert _shortest_len(adj, 1, 3, cap=5) == 2
    assert _shortest_len(adj, 0, 1, cap=5) == 1
    # cap: a path longer than cap reads as 0 (unreachable-or-farther)
    assert _shortest_len(adj, 0, 4, cap=3) == 0
    # genuinely unreachable
    disconnected = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
    assert _shortest_len(disconnected, 0, 3, cap=5) == 0


def test_sample_shapes_and_labels_in_range():
    task = GraphPlanningTask(n_nodes=7, ood_n_nodes=10, edge_prob=0.4, max_len=4)
    assert task.feature_dim == _MAX_NODES * _MAX_NODES + 2 * _MAX_NODES
    assert task.n_classes == 5
    b = task.sample(np.random.default_rng(0), 200, "id")
    assert b.x.shape == (200, task.feature_dim)
    assert b.y.shape == (200,)
    assert b.y.min() >= 0 and b.y.max() < task.n_classes


def test_encoded_adjacency_symmetric_and_onehots():
    task = GraphPlanningTask(n_nodes=7)
    b = task.sample(np.random.default_rng(1), 5, "id")
    adj_block = _MAX_NODES * _MAX_NODES
    for i in range(5):
        adj = b.x[i, :adj_block].reshape(_MAX_NODES, _MAX_NODES)
        assert np.allclose(adj, adj.T)                # undirected
        src = b.x[i, adj_block:adj_block + _MAX_NODES]
        dst = b.x[i, adj_block + _MAX_NODES:]
        assert src.sum() == 1.0 and dst.sum() == 1.0  # one source, one target
        assert int(src.argmax()) != int(dst.argmax())  # distinct


def test_ood_uses_more_nodes_same_feature_dim():
    task = GraphPlanningTask(n_nodes=7, ood_n_nodes=10)
    id_b = task.sample(np.random.default_rng(2), 300, "id")
    ood_b = task.sample(np.random.default_rng(2), 300, "ood")
    assert id_b.x.shape[1] == ood_b.x.shape[1] == task.feature_dim
    # OOD graphs use more nodes, so on average more edges are active
    assert task.difficulty(ood_b).mean() > task.difficulty(id_b).mean()


def test_determinism_and_registration():
    task = build_task("graph_planning", n_nodes=7, edge_prob=0.4)
    assert isinstance(task, GraphPlanningTask)
    a = task.sample(np.random.default_rng(3), 50, "id")
    b = task.sample(np.random.default_rng(3), 50, "id")
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
