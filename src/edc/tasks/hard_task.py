"""Hard 9x9 Sudoku (IRED benchmark) — the stronger-result task. [Phase 4]

Board encoded into ``x``; the reasoner descends to a full-grid latent decoded to digits. OOD
split = harder puzzles (fewer givens) than trained on. This is the task most likely to sit in
the partially-competent (~70-85%) regime where selective prediction has error to remove.
"""

from __future__ import annotations


class HardSudokuTask:
    name = "hard_task"

    def __init__(self, **kwargs) -> None:
        raise NotImplementedError("Phase 4: hard Sudoku generator + grid decoder.")
