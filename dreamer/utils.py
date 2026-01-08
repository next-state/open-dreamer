import jax
import jax.numpy as jnp
from flax import nnx
from dreamer.data import patchify, unpatchify
import orbax.checkpoint as ocp
from pathlib import Path
import optax
import grain
import operator
from einops import rearrange
from enum import IntEnum
from typing import Tuple, TypeVar
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from dreamer.configs import CheckpointConfig, LRScheduleConfig, OptimizerConfig


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
    return rearrange(z_btLd, '... (n_spatial k) d -> ... n_spatial (k d)', n_spatial=n_spatial, k=k)

def unpack_spatial_to_bottleneck(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,S_z, D_z_pre) -> (B,T,N_b,D_b) by splitting D_z_pre into k channels along N_b.
    Requires: N_b == n_spatial * k  (e.g., 256 -> 512 with k=2).
    """
    return rearrange(z_btLd, '... n_spatial (k d) -> ... (n_spatial k) d', n_spatial=n_spatial, k=k)


# ============================================================================
# Checkpointing Utilities
# ============================================================================

def build_checkpoint_manager(
        ckpt_cfg: CheckpointConfig,
        ckpt_dir: Path,
        item_names=("model_state", "optimizer_state", "train_dataloader_state", "rngs", "meta"),
    ) -> ocp.CheckpointManager:
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=ckpt_cfg.max_to_keep,
        save_interval_steps=ckpt_cfg.save_interval_steps,
        save_on_steps=[ckpt_cfg.max_steps - 1],  # always save at the end
    )

    return ocp.CheckpointManager(ckpt_dir, options=checkpoint_options, item_names=item_names)


ModelT = TypeVar('ModelT', bound=nnx.Module)


def try_restore(
        checkpoint_manager: ocp.CheckpointManager,
        model: ModelT,
        optimizer: nnx.Optimizer,
        train_iterator: grain.DataLoaderIterator,
        rngs: jax.Array,
    ) -> tuple[
        int, ModelT, nnx.Optimizer, grain.DataLoaderIterator, jax.Array
    ]:

    step = checkpoint_manager.latest_step()
    if step is None:
        print("No checkpoint found, starting from scratch.")
        return 0, model, optimizer, train_iterator, rngs
        
    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    
    restore_args = ocp.args.Composite(
        model_state=ocp.args.StandardRestore(model_state),  # type: ignore
        optimizer_state=ocp.args.StandardRestore(optimizer_state),  # type: ignore
        train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
        rngs=ocp.args.StandardRestore({"key": rngs}),  # type: ignore
    )
    
    restored = checkpoint_manager.restore(step, args=restore_args)
    nnx.update(model, restored["model_state"])
    nnx.update(optimizer, restored["optimizer_state"])
    train_iterator = restored["train_dataloader_state"]
    rngs = restored["rngs"]["key"]
    print(f"Restored checkpoint from step {step}.")

    return step + 1, model, optimizer, train_iterator, rngs


def maybe_save(
        checkpoint_manager: ocp.CheckpointManager,
        step: int,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        train_iterator: grain.DataLoaderIterator,
        rngs: jax.Array,
        meta: dict,
    ) -> None:
    if not checkpoint_manager.should_save(step):
        return

    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)

    checkpoint_manager_args = ocp.args.Composite(
        model_state=ocp.args.StandardSave(model_state),  # type: ignore
        optimizer_state=ocp.args.StandardSave(optimizer_state),  # type: ignore
        train_dataloader_state=grain.checkpoint.CheckpointSave(train_iterator),  # type: ignore
        rngs=ocp.args.StandardSave({'key': rngs}),  # type: ignore
        meta=ocp.args.JsonSave(meta)  # type: ignore
    )
    checkpoint_manager.save(step, args=checkpoint_manager_args)

# -------- Training utilities (shared across scripts) --------

def _ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_training_directories(cfg) -> tuple[Path, Path, Path, dict]:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    meta = {"cfg": OmegaConf.to_container(cfg, resolve=True)}
    print(f"[setup] output dir: {run_dir.resolve()}")
    return run_dir, ckpt_dir, vis_dir, meta


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


def build_lr_schedule(schedule_cfg: LRScheduleConfig) -> optax.Schedule:
    """
    Build learning rate schedule.
    
    Args:
        schedule_cfg: ScheduleConfig instance
    
    Returns:
        optax.Schedule instance
    """
    max_steps = schedule_cfg.max_steps
    if schedule_cfg.schedule_type == "constant":
        return optax.constant_schedule(value=schedule_cfg.lr)
    elif schedule_cfg.schedule_type == "cos":
        assert schedule_cfg.warmup_steps <= max_steps, "Warmup steps can't be greater than total steps."
        return optax.warmup_cosine_decay_schedule(
            init_value=schedule_cfg.init_lr,
            peak_value=schedule_cfg.lr,
            warmup_steps=schedule_cfg.warmup_steps,
            # Note: decay_steps includes the warmup steps, so pass the total value.
            decay_steps=max_steps,
            end_value=schedule_cfg.lr_end,
        )
    elif schedule_cfg.schedule_type == "wsd":
        assert (
            schedule_cfg.warmup_steps + schedule_cfg.wsd_decay_steps <= max_steps
        ), "Warmup and decay period is longer than total steps."
        schedules = [
            optax.linear_schedule(
                init_value=schedule_cfg.init_lr, end_value=schedule_cfg.lr, transition_steps=schedule_cfg.warmup_steps
            ),
            optax.constant_schedule(value=schedule_cfg.lr),
            optax.linear_schedule(
                init_value=schedule_cfg.lr, end_value=schedule_cfg.lr_end, transition_steps=schedule_cfg.wsd_decay_steps
            ),
        ]
        boundaries = [schedule_cfg.warmup_steps, max_steps - schedule_cfg.wsd_decay_steps]
        return optax.join_schedules(schedules, boundaries)
    else:
        raise ValueError(
            f"Unsupported learning rate schedule: {schedule_cfg.schedule_type}"
        )


def build_optimizer(optimizer_cfg: OptimizerConfig, model: nnx.Module, lr_schedule: optax.Schedule) -> nnx.Optimizer:
    """
    Build optimizer with given learning rate schedule.
    
    Args:
        optimizer_cfg: OptimizerConfig instance
        model: nnx.Module to optimize
        lr_schedule: optax.Schedule instance

    Returns:
        nnx.Optimizer instance
    """
    if optimizer_cfg.optimizer_type == "adamw":
        tx = optax.adamw(
            lr_schedule,
            b1=optimizer_cfg.b1,
            b2=optimizer_cfg.b2,
            weight_decay=optimizer_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_cfg.optimizer_type}")
    
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer
