"""Propositional logic / SAT-style reasoning family (port of EBRM). [Phase 4]

Clauses encoded into fixed-size ``x``; answer is satisfiability or a variable assignment.
OOD split = more variables/clauses than trained on.
"""

from __future__ import annotations


class LogicTask:
    name = "logic"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError("Phase 4: port EBRM logic_reasoning.jl generator.")
