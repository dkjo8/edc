"""Graph shortest-path / connectivity planning family (port of EBRM). [Phase 4]

Fixed-size adjacency + source/target encoded into ``x``; answer is a path/next-hop/length
label. OOD split = larger graphs than trained on (IRED-style size generalisation).
"""

from __future__ import annotations


class GraphPlanningTask:
    name = "graph_planning"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError("Phase 4: port EBRM graph_reasoning.jl generator.")
