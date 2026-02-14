import math
import einops
import jax
import jax.numpy as jnp
from typing import Any, Tuple
from .models import KVCachesDict, Dynamics, PolicyHeadMTP, Tokenizer
from .actions import Actions
from .utils import normalize_latents, unnormalize_latents
from flax.struct import dataclass

    
# ---------------------------
# Single-step τ-ladder denoiser
# ---------------------------

def next_latent(
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,                   # (B, 1, n_spatial, D_s)
    rng: jax.Array,
    prefill_length: int | None = None,
    task_embedding: jax.Array | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: KVCachesDict | None = None,
    latents_ctx: jax.Array| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: Actions | None = None,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, jax.Array]:
    raise NotImplementedError

def next_frame(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,
    dynamics_cache: Any,
    tokenizer_cache: Any,
    rng: jax.Array,
    task_embedding: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, Any, jax.Array]:
    raise NotImplementedError

def latent_rollout(
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    latents_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    deterministic: bool = False,
):
    raise NotImplementedError

def video_rollout(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    frames_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
):
    raise NotImplementedError
