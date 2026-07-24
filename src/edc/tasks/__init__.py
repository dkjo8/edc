"""Task families. Importing this package registers every built task.

Phase 1 ships ``arithmetic``. ``graph_planning``, ``logic``, and ``hard_task`` are stubbed
(Phase 4) and import lazily without erroring so the registry can list them.
"""

from edc.tasks import arithmetic  # noqa: F401  (side effect: registers "arithmetic")

# Phase 4 task families are intentionally not imported here until implemented.
__all__ = ["arithmetic"]
