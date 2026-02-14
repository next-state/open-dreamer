#!/usr/bin/env python3
"""
Test script for generation.py: loads a checkpoint and runs video rollouts on real data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_generation.py --checkpoint_dir /path/to/checkpoint [--config config.yaml] [--num_rollouts 2] [--horizon 4]
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import yaml

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.generation import DenoiseSchedule, video_rollout
from dreamer.parallel import build_parallel
from dreamer.data import make_dual_iterators, DatasetConfig


def test_video_rollout(
    checkpoint_dir: str,
    dataset_cfg: DatasetConfig,
    num_rollouts: int = 2,
    horizon: int = 4,
    output_dir: str = "test_outputs",
):
    """Test video_rollout with real data from the dataset."""
    print("\n" + "=" * 60)
    print("Testing video_rollout with real video data from dataset")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_dir}...")
    mesh, _, mesh_rules = build_parallel("data")

    with jax.set_mesh(mesh):
        bundle = DynamicsCheckpointBundle.from_pretrained(
            checkpoint_dir, mesh_rules=mesh_rules
        )
        dynamics = bundle.dynamics
        tokenizer = bundle.tokenizer

        # Get dimensions from config
        k_max = dynamics.cfg.k_max
        schedule = DenoiseSchedule.init(n_steps=4, k_max=k_max)

        # Create data loader
        print(f"Loading dataset from {dataset_cfg.data_dir}...")
        short_loader, _ = make_dual_iterators(
            dataset_cfg,
            short_T=dataset_cfg.T,
            long_T=dataset_cfg.T,
            num_workers=0,  # Single worker for testing
        )

        # Get one batch
        iterator = iter(short_loader)
        batch = next(iterator)

        videos = batch["videos"]  # (B, T, H, W, C)
        actions = batch["actions"]  # Actions pytree

        B, T, H, W, C = videos.shape
        print(f"Loaded batch: videos shape {videos.shape}, actions shape")
        print(f"  Binary actions: {actions.binary.shape}")
        print(f"  Categorical actions: {actions.categorical.shape}")
        print(f"  Continuous actions: {actions.continuous.shape}")

        # Split into context and future
        T_ctx = min(4, T - horizon)
        if T_ctx + horizon > T:
            print(f"Warning: T={T} is too small for T_ctx={T_ctx} + horizon={horizon}")
            T_ctx = T - horizon

        frames_ctx = videos[:, :T_ctx]
        actions_ctx = jax.tree.map(lambda x: x[:, :T_ctx] if x is not None else None, actions)

        print(f"\nContext: {T_ctx} frames")
        print(f"Horizon: {horizon} frames")
        print(f"Frame size: {H}x{W}x{C}")

        # Test video rollout
        print("\nRunning video_rollout...")
        for rollout_idx in range(num_rollouts):
            rng = jax.random.PRNGKey(rollout_idx)
            result = video_rollout(
                tokenizer=tokenizer,
                dynamics=dynamics,
                policy=actions[:, T_ctx : T_ctx + horizon],  # Use ground-truth actions
                schedule=schedule,
                frames_ctx=frames_ctx,
                actions_ctx=actions_ctx,
                num_steps=horizon,
                rng=rng,
                initial_task_embedding=None,
            )

            pred_frames = result["frames"]
            pred_latents = result["latents"]
            print(f"  Rollout {rollout_idx + 1}:")
            print(f"    Input frames shape: {frames_ctx.shape}")
            print(f"    Output frames shape: {pred_frames.shape}")
            print(f"    Output latents shape: {pred_latents.shape}")

            # Verify shapes
            assert (
                pred_frames.shape[0] == B
            ), f"Batch size mismatch: {pred_frames.shape[0]} vs {B}"
            assert (
                pred_frames.shape[1] == T_ctx + horizon
            ), f"Time steps mismatch: {pred_frames.shape[1]} vs {T_ctx + horizon}"
            assert (
                pred_frames.shape[2:] == (H, W, C)
            ), f"Frame size mismatch: {pred_frames.shape[2:]} vs {(H, W, C)}"
            print(f"    ✓ Shape checks passed")

            # Save visualization (first sample in batch, context + generated)
            try:
                import matplotlib.pyplot as plt

                sample_idx = 0
                fig, axes = plt.subplots(
                    1, T_ctx + horizon, figsize=(3 * (T_ctx + horizon), 3)
                )
                for t in range(T_ctx + horizon):
                    frame = pred_frames[sample_idx, t].astype(jnp.uint8)
                    axes[t].imshow(jnp.clip(frame, 0, 255))
                    axes[t].set_title(
                        f"t={t}" + (" (ctx)" if t < T_ctx else " (gen)")
                    )
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
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to dataset config file (YAML)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=2, help="Number of rollouts to test"
    )
    parser.add_argument(
        "--horizon", type=int, default=4, help="Number of future frames to generate"
    )
    parser.add_argument(
        "--output_dir", default="test_outputs", help="Directory to save visualizations"
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: checkpoint directory {checkpoint_dir} not found")
        return 1

    # Load dataset config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file {config_path} not found")
        return 1

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Extract dataset config
    dataset_cfg_dict = cfg_dict.get("dataset", {})
    dataset_cfg = DatasetConfig(**dataset_cfg_dict)

    print(f"\n{'='*60}")
    print("Generation.py Test Suite")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Dataset config: {config_path}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Horizon: {args.horizon}")
    print(f"Output directory: {args.output_dir}")

    try:
        test_video_rollout(
            str(checkpoint_dir),
            dataset_cfg,
            num_rollouts=args.num_rollouts,
            horizon=args.horizon,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
