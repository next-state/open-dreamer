"""
Chunked rollout evaluation for debugging autoregressive drift.

Produces a 3-column video: [Ground Truth | Normal Rollout | Chunked Rollout]

The chunked rollout splits a long sequence into shorter sub-sequences.
Each sub-sequence is a completely fresh rollout (new caches) initialized
with the last `ctx_len` predicted frames from the previous chunk.
"""
import logging
import types
from pathlib import Path

import hydra
import imageio.v3 as iio
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from omegaconf import OmegaConf

from dreamer.actions import shift_actions, Actions
from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import make_iterator
from dreamer.generation import DenoiseSchedule, latent_rollout
from dreamer.parallel import build_parallel
from dreamer.sampler import sample_video, decode_latents
from dreamer.utils import apply_border

logging.getLogger('absl').setLevel(logging.WARNING)

OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
OmegaConf.register_new_resolver("floordiv", lambda x, y: x // y)
OmegaConf.register_new_resolver("max", lambda *args: max(args))

jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@nnx.jit
def encode_jit(tokenizer, frames):
    latents, _ = tokenizer.encode(frames, deterministic=True)
    return latents


def chunked_latent_rollout(
    dynamics,
    tokenizer,
    all_latents: jax.Array,
    original_actions: Actions,
    categorical_action_dim: int,
    schedule: DenoiseSchedule,
    rng: jax.Array,
    ctx_len: int = 4,
    chunk_size: int = 64,
) -> jax.Array:
    """
    Run a chunked autoregressive rollout, re-initializing caches every `chunk_size` frames.

    Args:
        dynamics: Dynamics model.
        tokenizer: Tokenizer model (unused here, latents already encoded).
        all_latents: (B, T, S, D) full sequence of GT latents.
        original_actions: (B, T) UNSHIFTED actions from the dataset.
        categorical_action_dim: For shift_actions no-op creation.
        schedule: DenoiseSchedule for the dynamics model.
        rng: JAX random key.
        ctx_len: Number of context frames per chunk.
        chunk_size: Number of NEW frames to generate per chunk.

    Returns:
        (B, T, S, D) latents: context GT latents + all predicted latents concatenated.
    """
    B, T, S, D = all_latents.shape
    horizon = T - ctx_len  # total frames to generate

    # Collect all predicted latent chunks
    predicted_chunks = []

    num_chunks = (horizon + chunk_size - 1) // chunk_size
    for k in range(num_chunks):
        rollout_start = ctx_len + k * chunk_size
        frames_remaining = T - rollout_start
        this_chunk_size = min(chunk_size, frames_remaining)

        # --- Context latents ---
        if k == 0:
            ctx_latents = all_latents[:, :ctx_len]
        else:
            # Last ctx_len predicted frames from accumulation so far
            all_predicted = jnp.concatenate(predicted_chunks, axis=1)
            ctx_latents = all_predicted[:, -ctx_len:]

        # --- Actions: slice original (unshifted) actions, then shift ---
        ctx_start = rollout_start - ctx_len
        action_end = rollout_start + this_chunk_size
        # Slice covers ctx_len + this_chunk_size frames
        chunk_actions = original_actions[:, ctx_start:action_end]
        # Shift prepends no-op at position 0, drops last action
        chunk_actions_shifted = shift_actions(chunk_actions, categorical_action_dim)
        # Split into context and future
        actions_ctx = chunk_actions_shifted[:, :ctx_len]
        actions_future = chunk_actions_shifted[:, ctx_len:]

        print(f"  Chunk {k}/{num_chunks}: generating frames {rollout_start}..{rollout_start + this_chunk_size - 1} "
              f"(ctx from {'GT' if k == 0 else 'predicted'})")

        rng, chunk_rng = jax.random.split(rng)
        result = latent_rollout(
            dynamics,
            policy=actions_future,
            schedule=schedule,
            latents_ctx=ctx_latents,
            actions_ctx=actions_ctx,
            num_steps=this_chunk_size,
            rng=chunk_rng,
            deterministic=True,
        )

        # result['latents'] is (B, ctx_len + this_chunk_size, S, D)
        # We only want the NEW predicted frames (skip the context prefix)
        new_latents = result['latents'][:, ctx_len:]
        predicted_chunks.append(new_latents)

    # Concatenate: GT context + all predicted chunks
    all_predicted = jnp.concatenate(predicted_chunks, axis=1)
    out_latents = jnp.concatenate([all_latents[:, :ctx_len], all_predicted], axis=1)
    return out_latents


def run(cfg):
    rng = jax.random.PRNGKey(cfg.seed)
    data_seed = cfg.seed if cfg.dataset_seed is None else int(cfg.dataset_seed)

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        print(f"Loading checkpoint from: {cfg.dynamics_ckpt}")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            cfg.dynamics_ckpt, mesh_rules=mesh_rules,
            model_names={"dynamics", "dynamics_ema", "tokenizer"}
        )
        dynamics = bundle.dynamics_ema
        tokenizer = bundle.tokenizer
        print(f"Loaded dynamics (k_max={dynamics.cfg.k_max}, depth={dynamics.cfg.depth})")

        use_latent_data = cfg.dataset.data_type == "latent"
        ctx_len = int(cfg.eval_prompt_length)
        chunk_size = int(cfg.chunk_size)
        shortcut_steps = int(cfg.eval_shortcut_steps)
        k_max = dynamics.cfg.k_max
        schedule = DenoiseSchedule.init(shortcut_steps, k_max)
        categorical_action_dim = int(cfg.dataset.categorical_action_dim)

        # Load data
        print(f"Loading data from: {cfg.dataset.array_record_path}")
        iterator = make_iterator(cfg.dataset, seed=data_seed, device=data_sharding)
        batch = next(iter(iterator))

        original_actions = batch["actions"]  # unshifted
        shifted_actions = shift_actions(original_actions, categorical_action_dim)
        input_tensor = batch.get("latents") if use_latent_data else batch.get("videos")
        B, T = input_tensor.shape[:2]
        horizon = T - ctx_len
        print(f"Data: shape={input_tensor.shape}, ctx_len={ctx_len}, chunk_size={chunk_size}, horizon={horizon}")

        # --- Encode to latents if needed ---
        if use_latent_data:
            all_latents = input_tensor.astype(jnp.bfloat16)
        else:
            print("Encoding frames to latents...")
            all_latents = encode_jit(tokenizer, input_tensor)
            all_latents = all_latents.astype(jnp.bfloat16)

        # --- 1. Normal (full) rollout ---
        print("\n=== Normal rollout ===")
        rng, normal_rng = jax.random.split(rng)
        if use_latent_data:
            normal_pred, gt_decoded, _, _ = sample_video(
                tokenizer, dynamics, frames=None,
                actions=shifted_actions, horizon=horizon, schedule_config=schedule,
                rng=normal_rng, latents=input_tensor,
            )
            gt_frames = gt_decoded
        else:
            normal_pred, _, original_frames, _ = sample_video(
                tokenizer, dynamics, frames=input_tensor,
                actions=shifted_actions, horizon=horizon, schedule_config=schedule,
                rng=normal_rng,
            )
            gt_frames = original_frames
        print(f"Normal rollout done: pred shape={normal_pred.shape}")

        # --- 2. Chunked rollout ---
        print(f"\n=== Chunked rollout (chunk_size={chunk_size}) ===")
        rng, chunked_rng = jax.random.split(rng)
        chunked_latents = chunked_latent_rollout(
            dynamics=dynamics,
            tokenizer=tokenizer,
            all_latents=all_latents,
            original_actions=original_actions,
            categorical_action_dim=categorical_action_dim,
            schedule=schedule,
            rng=chunked_rng,
            ctx_len=ctx_len,
            chunk_size=chunk_size,
        )
        print(f"Chunked rollout done: latents shape={chunked_latents.shape}")

        # Decode chunked latents to frames
        print("Decoding chunked latents...")
        chunked_frames = decode_latents(tokenizer, chunked_latents)
        chunked_frames = jnp.clip(chunked_frames, 0, 255).astype(jnp.uint8)

        # --- 3. Visualization ---
        print("\n=== Creating visualization ===")
        num_videos = min(4, B)

        # Apply green border to context frames
        gt_vis = gt_frames.at[:, :ctx_len].set(
            apply_border(gt_frames[:, :ctx_len], color=(0, 255, 0))
        )
        normal_vis = normal_pred.at[:, :ctx_len].set(
            apply_border(normal_pred[:, :ctx_len], color=(0, 255, 0))
        )
        chunked_vis = chunked_frames.at[:, :ctx_len].set(
            apply_border(chunked_frames[:, :ctx_len], color=(0, 255, 0))
        )

        # Stack columns: [GT, Normal, Chunked]
        grid_columns = jnp.stack([gt_vis, normal_vis, chunked_vis])  # (3, B, T, H, W, C)
        grid_columns = grid_columns[:, :num_videos]
        videos = rearrange(grid_columns, 'S B T H W C -> T (B H) (S W) C', B=num_videos)

        # Save
        output_dir = Path(cfg.output_dir)
        vis_dir = output_dir / "visualizations" / f"step_{cfg.step:06d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = vis_dir / "chunked_rollout_comparison.mp4"

        videos_np = jax.device_get(videos)
        iio.imwrite(str(mp4_path), videos_np, fps=20, plugin='pyav', codec='libx264')
        print(f"\nVideo saved to: {mp4_path.resolve()}")
        print(f"Layout: [Ground Truth | Normal Rollout | Chunked Rollout (chunk={chunk_size})]")


@hydra.main(version_base=None, config_path="../configs", config_name="eval_dynamics")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
