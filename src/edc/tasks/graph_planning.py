"""Graph shortest-path planning family. [Phase 4d]

A relational/combinatorial reasoning family, deliberately distinct from additive arithmetic, to
test whether the geometry-beats-energy discovery *generalizes*. Each problem is a random undirected
Erdős–Rényi graph with a random ``(source, target)`` pair; the answer is the shortest-path length
(BFS), capped: classes ``1..max_len`` are the exact length, class ``0`` = unreachable-or-farther.

Features are fixed-size (padded to ``max_nodes``) so the in-distribution split (``n_nodes``) and the
OOD split (``ood_n_nodes`` > it — IRED-style size generalization) share one model:
flattened ``max_nodes × max_nodes`` adjacency ⊕ source one-hot ⊕ target one-hot.

Pure NumPy (invariant 1); BFS is a small per-sample loop (graphs are tiny).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from edc.registry import register_task
from edc.tasks.base import Batch

_MAX_NODES = 12  # padding capacity; both ID and OOD node counts must be <= this


def _shortest_len(adj: np.ndarray, src: int, dst: int, cap: int) -> int:
    """BFS shortest-path length in an undirected 0/1 adjacency; 0 if unreachable or > ``cap``."""
    if src == dst:
        return 0
    seen = {src}
    q = deque([(src, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= cap:
            continue
        for nbr in np.nonzero(adj[node])[0]:
            if nbr == dst:
                return dist + 1
            if nbr not in seen:
                seen.add(int(nbr))
                q.append((int(nbr), dist + 1))
    return 0  # unreachable-or-farther


class GraphPlanningTask:
    name = "graph_planning"

    def __init__(
        self,
        n_nodes: int = 7,
        ood_n_nodes: int = 10,
        edge_prob: float = 0.3,
        max_len: int = 4,
    ) -> None:
        if max(n_nodes, ood_n_nodes) > _MAX_NODES:
            raise ValueError(f"node count exceeds capacity {_MAX_NODES}")
        self.n_nodes = n_nodes
        self.ood_n_nodes = ood_n_nodes
        self.edge_prob = edge_prob
        self.max_len = max_len
        # class 0 = unreachable/farther, 1..max_len = exact shortest-path length.
        self.n_classes = max_len + 1
        self.feature_dim = _MAX_NODES * _MAX_NODES + 2 * _MAX_NODES

    def _n(self, split: str) -> int:
        return self.ood_n_nodes if split == "ood" else self.n_nodes

    def sample(self, rng: np.random.Generator, n: int, split: str = "id") -> Batch:
        k = self._n(split)
        x = np.zeros((n, self.feature_dim), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        adj_block = _MAX_NODES * _MAX_NODES
        for i in range(n):
            # Random undirected graph on the k active nodes (upper triangle -> symmetric).
            upper = (rng.random((k, k)) < self.edge_prob).astype(np.float32)
            upper = np.triu(upper, 1)
            adj = upper + upper.T
            src, dst = rng.choice(k, size=2, replace=False)
            y[i] = _shortest_len(adj, int(src), int(dst), self.max_len)

            padded = np.zeros((_MAX_NODES, _MAX_NODES), dtype=np.float32)
            padded[:k, :k] = adj
            x[i, :adj_block] = padded.reshape(-1)
            x[i, adj_block + int(src)] = 1.0                       # source one-hot
            x[i, adj_block + _MAX_NODES + int(dst)] = 1.0          # target one-hot
        return Batch(x=x, y=y)

    def evaluate(self, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(pred) == np.asarray(y)

    def difficulty(self, batch: Batch) -> np.ndarray:
        # Edge count (undirected) ~ graph density proxy.
        adj_block = _MAX_NODES * _MAX_NODES
        adj = batch.x[:, :adj_block]
        return (adj.sum(axis=1) / 2.0).astype(np.float32)


@register_task("graph_planning")
def _build(**kwargs) -> GraphPlanningTask:
    known = {"n_nodes", "ood_n_nodes", "edge_prob", "max_len"}
    return GraphPlanningTask(**{k: v for k, v in kwargs.items() if k in known})
