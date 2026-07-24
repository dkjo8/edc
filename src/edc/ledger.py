"""Append-only, sticky results ledger (JSONL).

One JSON object per run, appended to ``results/ledger.jsonl`` and never rewritten
(``.claude/CLAUDE.md`` invariant 5). Every figure and table is derived from this file,
so a run must record everything needed to interpret it: resolved config, git sha, env,
seed, and metrics. Analysis reads the *latest* row per ``(config_hash, seed)``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edc.config import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"


def config_hash(resolved: dict[str, Any]) -> str:
    """Stable short hash of a resolved config (order-independent)."""
    blob = json.dumps(resolved, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "nogit"
    except Exception:
        return "nogit"


def _env() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import jax  # local import keeps ledger usable without JAX

        info["jax"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
    except Exception:
        info["jax"] = None
    return info


@dataclass
class RunRecord:
    run_id: str
    timestamp: str  # caller-supplied (scripts stamp wall-clock; keeps this module pure)
    resolved_config: dict[str, Any]
    task: str
    split: str
    seed: int
    metrics: dict[str, Any]
    artifact_paths: list[str] | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": _git_sha(),
            "config_hash": config_hash(self.resolved_config),
            "config": self.resolved_config,
            "task": self.task,
            "split": self.split,
            "seed": self.seed,
            "metrics": self.metrics,
            "env": _env(),
            "artifact_paths": self.artifact_paths or [],
        }


def append(record: RunRecord, path: str | Path = LEDGER_PATH) -> dict[str, Any]:
    """Append one run record and return the written row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = record.to_row()
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def read_all(path: str | Path = LEDGER_PATH) -> list[dict[str, Any]]:
    """Read every ledger row (chronological)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_per_config(path: str | Path = LEDGER_PATH) -> dict[tuple[str, int], dict[str, Any]]:
    """Latest row per ``(config_hash, seed)`` — the canonical view for analysis."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_all(path):
        out[(row["config_hash"], row["seed"])] = row
    return out
