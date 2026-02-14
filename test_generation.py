#!/usr/bin/env python3
"""
Test script for generation.py: loads a checkpoint and runs a few rollouts.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_generation.py --checkpoint_dir /path/to/checkpoint [--num_rollouts 3] [--horizon 4]
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.generation import DenoiseSchedule, latent_rollout, video_rollout
from dreamer.parallel import build_parallel
from dreamer.actions import Actions


def normalize_to_uint8(x: jnp.ndarray) -> jnp.ndarray:
    """Normalize from [-1, 1] or [0, 1] to [0, 255] uint8."""
    x = jnp.clip(x, 0, 1) * 255
    return x.astype(jnp.uint8)


def compute_mse_normalized(pred: jnp.ndarray, target: jnp.ndarray) -> float:
    """Compute MSE in normalized pixel space (before uint8 conversion)."""
    # Assuming both are in [0, 1] range
    mse = jnp.mean((pred - target) ** 2)
    return float(mse)


def test_latent_rollout(checkpoint_dir: str, num_rollouts: int = 2, horizon: int = 4):
    """Test latent_rollout with dummy data."""
    print("\n" + "="*60)
    print("Testing latent_rollout with dummy data")
    print("="*60)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_dir}...")
    mesh, sharding, mesh_rules = build_parallel("data")  # Uses CUDA_VISIBLE_DEVICES
    bundle = DynamicsCheckpointBundle.from_pretrained(checkpoint_dir, mesh_rules=mesh_rules)
    dynamics = bundle.dynamics
    tokenizer = bundle.tokenizer

    # Get dimensions from config
    B = 2  # batch size
    T_ctx = 4  # context length
    n_latents = tokenizer.cfg.encoder.n_latents
    d_bottleneck = tokenizer.cfg.encoder.d_bottleneck

    print(f"Batch size: {B}, Context length: {T_ctx}, Horizon: {horizon}")
    print(f"n_latents: {n_latents}, d_bottleneck: {d_bottleneck}")

    # Create dummy latents (context)
    latents_ctx = jnp.ones((B, T_ctx, n_latents, d_bottleneck), dtype=jnp.bfloat16)
    print(f"Context latents shape: {latents_ctx.shape}")

    # Create dummy actions (context + future)
    # Actions has fields: binary, categorical, continuous
    categorical_action_dim = dynamics.cfg.categorical_action_dim
    actions_ctx = Actions(
        binary=jnp.ones((B, T_ctx, dynamics.cfg.num_binary_actions), dtype=jnp.int32),
        categorical=jnp.zeros((B, T_ctx), dtype=jnp.int32),
        continuous=jnp.zeros((B, T_ctx, dynamics.cfg.continuous_action_dim), dtype=jnp.float32),
    )
    actions_future = Actions(
        binary=jnp.ones((B, horizon, dynamics.cfg.num_binary_actions), dtype=jnp.int32),
        categorical=jnp.zeros((B, horizon), dtype=jnp.int32),
        continuous=jnp.zeros((B, horizon, dynamics.cfg.continuous_action_dim), dtype=jnp.float32),
    )
    print(f"Context actions shape: binary {actions_ctx.binary.shape}, categorical {actions_ctx.categorical.shape}")
    print(f"Future actions shape: binary {actions_future.binary.shape}, categorical {actions_future.categorical.shape}")

    # Create schedule (shortcut with K=4 steps)
    k_max = dynamics.cfg.k_max
    schedule = DenoiseSchedule.init(n_steps=4, k_max=k_max)
    print(f"Schedule: K={schedule.n_steps}, k_max={schedule.k_max}, tau_ctx={schedule.tau_ctx}")

    # Test rollout
    print("\nRunning latent_rollout...")
    for rollout_idx in range(num_rollouts):
        rng = jax.random.PRNGKey(rollout_idx)
        result = latent_rollout(
            dynamics=dynamics,
            policy=actions_future,  # Use ground-truth actions
            schedule=schedule,
            latents_ctx=latents_ctx,
            actions_ctx=actions_ctx,
            num_steps=horizon,
            rng=rng,
            initial_task_embedding=None,
            deterministic=True,
        )

        pred_latents = result["latents"]
        print(f"  Rollout {rollout_idx+1}: output shape {pred_latents.shape}")
        print(f"    Context: {T_ctx} frames, Generated: {horizon} frames")
        assert pred_latents.shape == (B, T_ctx + horizon, n_latents, d_bottleneck)
        print(f"    ✓ Shape check passed")

    print("\n✓ latent_rollout tests passed!")


def test_video_rollout(checkpoint_dir: str, num_rollouts: int = 2, horizon: int = 4, output_dir: str = "test_outputs"):
    """Test video_rollout with dummy data."""
    print("\n" + "="*60)
    print("Testing video_rollout with dummy video data")
    print("="*60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_dir}...")
    mesh, sharding, mesh_rules = build_parallel("data")  # Uses CUDA_VISIBLE_DEVICES
    bundle = DynamicsCheckpointBundle.from_pretrained(checkpoint_dir, mesh_rules=mesh_rules)
    dynamics = bundle.dynamics
    tokenizer = bundle.tokenizer

    # Get dimensions
    B = 2
    T_ctx = 4
    patch_size = tokenizer.cfg.encoder.patch_size
    H, W, C = patch_size * 4, patch_size * 4, 3

    print(f"Batch size: {B}, Context length: {T_ctx}, Horizon: {horizon}")
    print(f"Frame size: {H}x{W}x{C}")

    # Create dummy frames (context)
    frames_ctx = jnp.ones((B, T_ctx, H, W, C), dtype=jnp.uint8) * 128  # mid-gray
    print(f"Context frames shape: {frames_ctx.shape}")

    # Create dummy actions
    actions_ctx = Actions(
        binary=jnp.ones((B, T_ctx, dynamics.cfg.num_binary_actions), dtype=jnp.int32),
        categorical=jnp.zeros((B, T_ctx), dtype=jnp.int32),
        continuous=jnp.zeros((B, T_ctx, dynamics.cfg.continuous_action_dim), dtype=jnp.float32),
    )
    actions_future = Actions(
        binary=jnp.ones((B, horizon, dynamics.cfg.num_binary_actions), dtype=jnp.int32),
        categorical=jnp.zeros((B, horizon), dtype=jnp.int32),
        continuous=jnp.zeros((B, horizon, dynamics.cfg.continuous_action_dim), dtype=jnp.float32),
    )

    # Create schedule
    k_max = dynamics.cfg.k_max
    schedule = DenoiseSchedule.init(n_steps=4, k_max=k_max)

    # Test video rollout
    print("\nRunning video_rollout...")
    for rollout_idx in range(num_rollouts):
        rng = jax.random.PRNGKey(100 + rollout_idx)
        result = video_rollout(
            tokenizer=tokenizer,
            dynamics=dynamics,
            policy=actions_future,
            schedule=schedule,
            frames_ctx=frames_ctx,
            actions_ctx=actions_ctx,
            num_steps=horizon,
            rng=rng,
            initial_task_embedding=None,
        )

        pred_frames = result["frames"]
        pred_latents = result["latents"]
        print(f"  Rollout {rollout_idx+1}:")
        print(f"    Frames shape: {pred_frames.shape}")
        print(f"    Latents shape: {pred_latents.shape}")
        assert pred_frames.shape[0] == B
        assert pred_frames.shape[1] == T_ctx + horizon
        assert pred_frames.shape[-1] == C
        print(f"    ✓ Shape checks passed")

        # Compute MSE between predicted and target frames (normalized space)
        # Normalize frames to [0, 1]
        pred_norm = pred_frames.astype(jnp.float32) / 255.0
        target_norm = frames_ctx.astype(jnp.float32) / 255.0

        # MSE only on context (we can't compute MSE on future since we don't have GT)
        mse = compute_mse_normalized(pred_norm[:, :T_ctx], target_norm)
        print(f"    MSE (context, normalized space): {mse:.6f}")

        # Save visualization (first sample in batch, context + generated)
        try:
            import matplotlib.pyplot as plt

            sample_idx = 0
            fig, axes = plt.subplots(1, T_ctx + horizon, figsize=(3*(T_ctx + horizon), 3))
            for t in range(T_ctx + horizon):
                frame = pred_frames[sample_idx, t].astype(jnp.uint8)
                axes[t].imshow(jnp.clip(frame, 0, 255))
                axes[t].set_title(f"t={t}" + (" (ctx)" if t < T_ctx else " (gen)"))
                axes[t].axis("off")

            save_path = output_path / f"rollout_{rollout_idx}.png"
            plt.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close()
            print(f"    ✓ Saved visualization to {save_path}")
        except ImportError:
            print(f"    (matplotlib not available, skipping visualization)")

    print("\n✓ video_rollout tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test generation.py with a checkpoint")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--num_rollouts", type=int, default=2, help="Number of rollouts to test")
    parser.add_argument("--horizon", type=int, default=4, help="Number of future frames to generate")
    parser.add_argument("--output_dir", default="test_outputs", help="Directory to save visualizations")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: checkpoint directory {checkpoint_dir} not found")
        return 1

    print(f"\n{'='*60}")
    print("Generation.py Test Suite")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Horizon: {args.horizon}")
    print(f"Output directory: {args.output_dir}")

    try:
        # Test latent rollout (cheaper, no tokenizer encode/decode)
        test_latent_rollout(str(checkpoint_dir), num_rollouts=args.num_rollouts, horizon=args.horizon)

        # Test video rollout (more expensive, with tokenizer)
        test_video_rollout(
            str(checkpoint_dir),
            num_rollouts=args.num_rollouts,
            horizon=args.horizon,
            output_dir=args.output_dir
        )

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
