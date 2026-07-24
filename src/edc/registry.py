"""Name -> builder registries, so configs stay declarative (``task = "arithmetic"``).

Kept deliberately tiny and framework-agnostic. Tasks self-register on import; the
resolver imports the task package lazily to avoid pulling JAX into pure-Python paths.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_TASKS: dict[str, Callable[..., object]] = {}


def register_task(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def deco(builder: Callable[..., T]) -> Callable[..., T]:
        if name in _TASKS:
            raise ValueError(f"task '{name}' already registered")
        _TASKS[name] = builder
        return builder

    return deco


def build_task(name: str, **kwargs: object) -> object:
    if name not in _TASKS:
        # Import the task package so its @register_task decorators run, then retry.
        import importlib

        importlib.import_module("edc.tasks")
    if name not in _TASKS:
        raise KeyError(f"unknown task '{name}'. registered: {sorted(_TASKS)}")
    return _TASKS[name](**kwargs)


def available_tasks() -> list[str]:
    import importlib

    importlib.import_module("edc.tasks")
    return sorted(_TASKS)
