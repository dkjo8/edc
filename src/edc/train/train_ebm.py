"""Train the energy reasoner. Returns trained params + inference fns + history.

Small, CPU-friendly loop (optax Adam). Deterministic given ``cfg.run.seed``. This is the
Phase-1 workhorse used by ``edc.cli smoke`` and later by the experiment runners.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from edc.energy import mlp_ebm
from edc.seeding import fold, numpy_rng, root_key
from edc.train.losses import ired_loss_fn, loss_fn


def train(cfg, task):
    """Train on ``task``. Returns ``(params, fns, history)``.

    ``cfg.train.objective`` selects the objective: ``"basin_center"`` (Phase-1 supervised bowl,
    default) or ``"ired"`` (denoising score matching over a learned multi-basin landscape).
    """
    key = root_key(cfg.run.seed)
    key, k_build = jax.random.split(key)

    params, fns, model = mlp_ebm.build(cfg, task.n_classes, task.feature_dim, k_build)

    opt = optax.adam(cfg.train.lr)
    opt_state = opt.init(params)

    objective = getattr(cfg.train, "objective", "basin_center")
    if objective == "ired":
        nmin, nmax = cfg.train.ired_noise_min, cfg.train.ired_noise_max
        dw = cfg.train.ired_decode_weight

        init_scale = cfg.inference.init_scale

        def _loss(params, model, x, y, key):
            return ired_loss_fn(params, model, x, y, key, nmin, nmax, dw, init_scale)
    else:
        neg_noise = cfg.train.neg_noise

        def _loss(params, model, x, y, key):
            return loss_fn(params, model, x, y, key, neg_noise)

    grad_loss = jax.jit(jax.value_and_grad(_loss, has_aux=True), static_argnums=(1,))

    # One fixed training set (host-side, deterministic).
    rng = numpy_rng(cfg.run.seed, 1)
    data = task.sample(rng, cfg.train.n_train, split="id")
    x_all, y_all = jnp.asarray(data.x), jnp.asarray(data.y)

    n = cfg.train.n_train
    bs = cfg.train.batch_size
    history: list[dict] = []
    step = 0
    for epoch in range(cfg.train.epochs):
        order = np.asarray(numpy_rng(cfg.run.seed, 2, epoch).permutation(n))
        for i in range(0, n - bs + 1, bs):
            idx = order[i : i + bs]
            k_step = fold(key, step)
            (loss, metrics), grads = grad_loss(
                params, model, x_all[idx], y_all[idx], k_step
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            step += 1
        history.append({k: float(v) for k, v in metrics.items()})

    return params, fns, {"epochs": history}
