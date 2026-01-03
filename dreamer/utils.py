import jax
import jax.numpy as jnp
from flax import nnx
from dreamer.data import patchify, unpatchify
import orbax.checkpoint as ocp
from pathlib import Path
import optax
import operator
from einops import rearrange
from enum import IntEnum
from typing import Tuple
import numpy as np
import math


# --- dtype helpers ---

def to_jnp_dtype(dtype: str | jnp.dtype) -> jnp.dtype:
    """Convert string or jnp.dtype to jnp.dtype."""
    if isinstance(dtype, jnp.dtype):
        return dtype
    if dtype == "float32":
        return jnp.float32
    if dtype == "float16":
        return jnp.float16
    if dtype == "bfloat16":
        return jnp.bfloat16
    return jnp.dtype(dtype)


# --- math helpers ---

def is_pow2_frac(x: float) -> bool:
    """Check if x is a power-of-two fraction (1/2, 1/4, 1/8, etc.)."""
    if x <= 0 or x > 1:
        return False
    inv = round(1.0 / x)
    return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

# --- helpers ---
temporal_patchify = jax.jit(
    jax.vmap(patchify, in_axes=(1, None), out_axes=1),  # (B,T,H,W,C) -> (B,T,Np,Dp)
    static_argnames=("patch",),
)

temporal_unpatchify = jax.jit(
    jax.vmap(unpatchify, in_axes=(1, None, None, None), out_axes=1),
    static_argnames=("patch", "H", "W"),
)

class Modality(IntEnum):
    LATENT   = -1
    IMAGE    = 0
    ACTION   = 1
    PROPRIO  = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7
    # add more as needed

@jax.tree_util.register_pytree_node_class
class TokenLayout:
    """
    Ordered token layout for a single timestep: segments define the order.
    """
    def __init__(self, segments: Tuple[Tuple[Modality, int], ...]):
        self.segments = segments  # e.g. ((Modality.LATENT, n_latents), (Modality.IMAGE, n_patches), ...)

    @property
    def S(self) -> int:
        return sum(n for _, n in self.segments)

    def modality_ids(self) -> jnp.ndarray:
        parts = [jnp.full((n,), int(m), dtype=jnp.int32) for m, n in self.segments]
        return jnp.concatenate(parts)

    def slices(self) -> dict:
        """Convenience: start/stop indices per modality (first occurrence if repeated)."""
        idx = 0
        out = {}
        for m, n in self.segments:
            out[m] = slice(idx, idx + n)
            idx += n
        return out

    def make_mask(self, mode: str):
        """
        Returns a (1, 1, S, S) boolean mask indicating allowed key for each query index, per mode.
        S = number of tokens in a single frame.

        Modes:
        - "encoder":
            - Latent tokens (query) can attend to ALL tokens (key).
            - Non-latent tokens (query) can ONLY attend to tokens of the SAME modality (key).
        - "decoder":
            - Latent tokens (query) can ONLY attend to Latent tokens (key).
            - Non-latent tokens (query) can attend to tokens of the SAME modality AND Latent tokens (key).
        - "wm_agent":
            - Action tokens (query) can ONLY attend to Action tokens (key).
            - Observation tokens (query) can attend to Observation AND Action tokens (key).
            - Agent tokens (query) can attend to ALL tokens (key).
        """
        modality_ids = self.modality_ids()
        S = self.S

        # Broadcast helpers
        q_idx = jnp.arange(S)[:, None]       # (S, 1)
        k_idx = jnp.arange(S)[None, :]       # (1, S)

        q_mod = modality_ids[q_idx]      # (S, 1)
        k_mod = modality_ids[k_idx]      # (1, S)

        if mode == "encoder":
            # latents -> all; non-latents -> same modality only
            mask = (q_mod == k_mod) | (q_mod == Modality.LATENT)
        elif mode == "decoder":
            # latents -> latents only; non-latents -> same modality + latents
            mask = (q_mod == k_mod) | (k_mod == Modality.LATENT)
        elif mode == "wm_agent":
            # wm_agent:

            # Hierarchy levels: Action=0, Obs=1, Agent=2
            # mask = level(q) >= level(k)

            def get_level(mod):
                # Default to 1 (Obs)
                lvl = jnp.ones_like(mod, dtype=jnp.int32) # Default to 1 (Obs)
                lvl = jnp.where(mod == Modality.ACTION, 0, lvl) # Set to 0 if Action
                lvl = jnp.where(mod == Modality.AGENT, 2, lvl) # Set to 2 if Agent

                return lvl

            q_level = get_level(q_mod)
            k_level = get_level(k_mod)

            mask = q_level >= k_level
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Save (1, 1, S, S)
        mask = mask[None, None, :, :]
        mask = jax.lax.stop_gradient(mask)
        return mask

    def tree_flatten(self):
        return ((), self.segments)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


def normalize_with_dataset_stats(videos, *, mean, std):
    """
    Normalize videos using dataset-level statistics.

    Handles spatial images (B, T, H, W, C).
    For flattened patches, tiles the per-channel stats to match the interleaved layout.

    Args:
        videos: input videos/patches
        mean: dataset mean (list of C floats for per-channel)
        std: dataset std (list of C floats for per-channel)
    Returns:
        normalized videos
    """
    videos = videos.astype(jnp.float32)/255
    mean_arr = jnp.asarray(mean, dtype=videos.dtype)
    std_arr = jnp.asarray(std, dtype=videos.dtype)

    mean_c = jnp.expand_dims(mean_arr, axis=(0, 1, 2, 3))
    std_c =  jnp.expand_dims(std_arr, axis=(0, 1, 2, 3))

    return (videos - mean_c) / std_c

def unnormalize_with_dataset_stats(normalized_videos, *, mean, std):
    """
    Unnormalize videos using dataset-level statistics.

    Handles both flattened patches (B, T, N, patch*patch*C) and spatial images (B, T, H, W, C).
    For flattened patches, tiles the per-channel stats to match the interleaved layout.

    Args:
        normalized_videos: normalized videos/patches
        mean: dataset mean (list of C floats for per-channel)
        std: dataset std (list of C floats for per-channel)
    Returns:
        unnormalized videos
    """
    mean_arr = jnp.asarray(mean, dtype=normalized_videos.dtype)
    std_arr = jnp.asarray(std, dtype=normalized_videos.dtype)

    mean_c = jnp.expand_dims(mean_arr, axis=(0, 1, 2, 3))
    std_c =  jnp.expand_dims(std_arr, axis=(0, 1, 2, 3))

    return (normalized_videos * std_c + mean_c)*255

def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
    Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
    """
    return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)

def unpack_spatial_to_bottleneck(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,S_z, D_z_pre) -> (B,T,N_b,D_b) by splitting D_z_pre into k channels along N_b.
    Requires: N_b == n_spatial * k  (e.g., 256 -> 512 with k=2).
    """
    return rearrange(z_btLd, 'b t n_spatial (k d) -> b t (n_spatial k) d', n_spatial=n_spatial, k=k)


# ============================================================================
# Checkpointing Utilities
# ============================================================================

def make_state(model_or_dict, opt_state, rng, step):
    """
    Pack training state as a PyTree for checkpointing (JAX/Orbax-friendly types only).
    
    In NNX, we can checkpoint either:
    - nnx.state(model) -> gets all trainable params + batch stats
    - Just the parameters dict

    Args:
        model_or_dict: Either an NNX model or a dict of parameters
        opt_state: Optimizer state
        rng: Random key
        step: Current step

    Returns:
        State dict suitable for Orbax checkpointing
    """
    # If it's a dict, use directly; otherwise extract state from NNX model
    if isinstance(model_or_dict, dict):
        state = model_or_dict
    else:
        graphdef, *states = nnx.split(model_or_dict, nnx.Param, nnx.BatchStat, ...)
        state = nnx.State.merge(*states)

    return {
        "params": state,
        "opt_state": opt_state,
        "rng": rng,
        "step": jnp.int32(step),
    }


def make_manager(ckpt_dir: str | Path, max_to_keep: int = 5, save_interval_steps: int = 1000, item_names=("state","meta")):
    path = Path(ckpt_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep,
                                           save_interval_steps=save_interval_steps)
    # item_names gives nice attribute access on restore: restored.state, restored.meta
    mngr = ocp.CheckpointManager(path, options=options, item_names=item_names)
    return mngr


def try_restore(mngr: ocp.CheckpointManager, state_example: dict, ctx, meta: dict | None = None):
    """
    Build abstract trees from current shapes/dtypes so Orbax can restore safely.
    
    Creates abstract targets with sharding info from ctx so Orbax loads directly
    into GPU memory with proper sharding/replication.
    """
    # Create abstract targets WITH sharding info for direct GPU loading
    def to_sharded_abstract(x):
        if isinstance(x, jax.Array):
            # Params/opt_state are replicated across available devices
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=ctx.replicated_sharding)
        return ocp.utils.to_shape_dtype_struct(x)
    
    abstract_state = jax.tree_util.tree_map(to_sharded_abstract, state_example)
    
    restore_args = ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),
        meta=ocp.args.JsonRestore() if meta is not None else None
    )
    latest = mngr.latest_step()
    if latest is None:
        return None
    restored = mngr.restore(latest, args=restore_args)
    return latest, restored


def maybe_save(mngr: ocp.CheckpointManager, step: int, state: dict, meta: dict | None = None):
    if not mngr.should_save(step):  # obey save interval policy
        return
    save_args = ocp.args.Composite(
        state=ocp.args.StandardSave(state),
        meta=ocp.args.JsonSave(meta) if meta is not None else None
    )
    mngr.save(step, args=save_args)  # async by default; runs in a background thread


# ============================================================================
# Model Initialization Utilities
# ============================================================================

def init_tokenizer(rng, tokenizer_config):
    """
    Initialize a tokenizer model with NNX.
    
    Args:
        rng: JAX random key
        tokenizer_config: TokenizerConfig instance
        
    Returns:
        rng: Updated random key
        tokenizer: Initialized Tokenizer model
    """
    from dreamer.models import Tokenizer

    rng, model_rng = jax.random.split(rng)
    rngs = nnx.Rngs(model_rng)

    tokenizer = Tokenizer(tokenizer_config, rngs=rngs)

    return rng, tokenizer


def init_dynamics(rng, dynamics_config, tokenizer_config):
    """
    Initialize a dynamics model with NNX.
    
    Args:
        rng: JAX random key
        dynamics_config: DynamicsModelConfig instance
        tokenizer_config: TokenizerConfig instance (for spatial dims)
        
    Returns:
        rng: Updated random key
        dynamics: Initialized Dynamics model
    """
    from dreamer.models import Dynamics
    
    rng, model_rng = jax.random.split(rng)
    rngs = nnx.Rngs(model_rng)
    
    dynamics = Dynamics(dynamics_config, rngs=rngs)
    
    return rng, dynamics


# -------- Training utilities (shared across scripts) --------

def _ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    """Convert float32 image to uint8."""
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def apply_border(frames: jnp.ndarray, color = (255, 0, 0), width: int = 2) -> jnp.ndarray:
    """
    Add a colored border to a batch of frames.
    """
    color = jnp.asarray(color, dtype=frames.dtype)
    frames = frames.at[..., :width, :, :].set(color)
    frames = frames.at[..., -width:, :, :].set(color)
    frames = frames.at[..., :, :width, :].set(color)
    frames = frames.at[..., :, -width:, :].set(color)
    return frames

def from_dict(cls, d):
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        t = field_types[k]
        if hasattr(t, "__dataclass_fields__"):
            kwargs[k] = from_dict(t, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)

def recursive_list_to_tuple(d):
    if isinstance(d, list):
        return tuple(recursive_list_to_tuple(x) for x in d)
    if isinstance(d, dict):
        return {k: recursive_list_to_tuple(v) for k, v in d.items()}
    return d

def _count_component(component_params):
    """Count total parameters in a component."""
    params_sizes = jax.tree.map(jax.numpy.size, component_params)
    total_parameters = jax.tree.reduce(operator.add, params_sizes)
    return total_parameters


def count_parameters_by_component(model):
    """
    Count parameters for each component of an NNX model.

    Args:
        model: NNX Model instance

    Returns:
        Dictionary with parameter counts for each component
    """
    # Split model to get parameter structure
    graphdef, state, _ = nnx.split(model, nnx.Param, ...)

    # Count parameters for each top-level component
    counts = {}
    total_params = 0

    for name, component in state.items():
        count = _count_component(component)
        counts[name] = count
        total_params += count

    counts["total"] = total_params
    return counts


def get_lr_schedule(
    lr_schedule: str,
    init_lr: float,
    max_lr: float,
    decay_end: float,
    total_steps: int,
    warmup_steps: int,
    wsd_decay_steps: int,
) -> optax.Schedule:
    """
    Learning-rate schedule helper, mirrored from Jasmine.

    Supported schedules:
      - "cos": warmup cosine decay
      - "wsd": warmup -> hold -> decay (linear warmup, constant hold, linear decay)
    """
    supported_schedules = ["wsd", "cos"]
    if lr_schedule == "cos":
        assert warmup_steps <= total_steps, "Warmup steps can't be greater than total steps."
        return optax.warmup_cosine_decay_schedule(
            init_value=init_lr,
            peak_value=max_lr,
            warmup_steps=warmup_steps,
            # Note: decay_steps includes the warmup steps, so pass the total value.
            decay_steps=total_steps,
            end_value=decay_end,
        )
    elif lr_schedule == "wsd":
        assert (
            warmup_steps + wsd_decay_steps <= total_steps
        ), "Warmup and decay period is longer than total steps."
        schedules = [
            optax.linear_schedule(
                init_value=init_lr, end_value=max_lr, transition_steps=warmup_steps
            ),
            optax.constant_schedule(value=max_lr),
            optax.linear_schedule(
                init_value=max_lr, end_value=decay_end, transition_steps=wsd_decay_steps
            ),
        ]
        boundaries = [warmup_steps, total_steps - wsd_decay_steps]
        return optax.join_schedules(schedules, boundaries)
    else:
        raise ValueError(
            f"Learning rate schedule not supported. Please use one of {supported_schedules}"
        )
