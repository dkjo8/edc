"""Modal GPU runner (mirrors EBRM's modal_run.py). [Phase 4]

Defines a Modal app with a pinned CUDA-JAX image, mounts src/ + configs/, runs an experiment
or sweep on a GPU box, and syncs results/ledger.jsonl back. All CPU-only work stays local; this
exists purely to scale the full experiment suite. Import is guarded so the repo installs without
Modal present.
"""

from __future__ import annotations

# NOTE: intentionally not importing `modal` at module load — keep the offline env clean.
PLACEHOLDER = "Phase 4: define Modal app (image=jax[cuda12], mounts, run function)."
