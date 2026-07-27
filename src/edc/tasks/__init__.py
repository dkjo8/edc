"""Task families. Importing this package registers every built task.

Phase 1 ships ``arithmetic``; Phase 4d adds ``graph_planning``. ``logic`` and ``hard_task`` remain
stubbed and are not imported until implemented.
"""

from edc.tasks import (  # noqa: F401  (side effects: register the tasks)
    arithmetic,
    graph_planning,
)

__all__ = ["arithmetic", "graph_planning"]
