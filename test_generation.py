#!/usr/bin/env python3
"""
Test script for generation.py: loads a checkpoint and runs video rollouts.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_generation.py --checkpoint_dir /path/to/checkpoint [--num_rollouts 2] [--horizon 4]
"""

import argparse
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
from tqdm import tqdm

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.generation import DenoiseSchedule, video_rollout
from dreamer.parallel import build_parallel
from dreamer.actions import Actions
from dreamer.data import make_dual_iterators


def test_video_rollout(
    checkpoint_dir: str,
    num_rollouts: int = 2,
    horizon: int = 4,
    output_dir: str = "test_outputs",
):
    """Test video_rollout with real frames from dataset."""
    print("\n" + "=" * 60)
    print("Testing video_rollout")
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

        # Get dimensions from checkpoint
        H = tokenizer.cfg.decoder.H
        W = tokenizer.cfg.decoder.W
        C = 3  # RGB
        T_ctx = 4

        print(f"Frame dimensions from checkpoint: {H}x{W}x{C}")
        print(f"Context length: {T_ctx}, Horizon: {horizon}")

        # Get dynamics config
        k_max = dynamics.cfg.k_max
        schedule = DenoiseSchedule.init(n_steps=4, k_max=k_max)

        # Load real data from dataset
        print(f"Loading dataset...")
        dataset_cfg = dynamics.cfg.dataset

        short_loader, _ = make_dual_iterators(
            dataset_cfg,
            short_T=T_ctx + horizon + 4,  # Extra buffer
            long_T=T_ctx + horizon + 4,
            num_workers=0,
        )

        iterator = iter(short_loader)
        batch = next(iterator)

        videos = batch["videos"]  # (B, T, H, W, C)
        actions = batch["actions"]  # Actions pytree

        # Take first few frames as context, next horizon frames for generation
        T_total = min(T_ctx + horizon, videos.shape[1] - 1)
        frames_all = videos[:, :T_total]
        actions_all = jax.tree.map(lambda x: x[:, :T_total] if x is not None else None, actions)

        frames_ctx = frames_all[:, :T_ctx]
        frames_future = frames_all[:, T_ctx : T_ctx + horizon]

        actions_ctx = jax.tree.map(lambda x: x[:, :T_ctx] if x is not None else None, actions_all)
        actions_future = jax.tree.map(lambda x: x[:, T_ctx : T_ctx + horizon] if x is not None else None, actions_all)

        B = frames_ctx.shape[0]

        print(f"Loaded batch: videos {videos.shape}, B={B}")
        print(f"Context frames shape: {frames_ctx.shape}")
        if actions_ctx.binary is not None:
            print(f"Context actions: binary {actions_ctx.binary.shape}, categorical {actions_ctx.categorical.shape}")
        if actions_future.binary is not None:
            print(f"Future actions: binary {actions_future.binary.shape}, categorical {actions_future.categorical.shape}")

        # Test video rollout
        print("\nRunning video_rollout...")
        for rollout_idx in tqdm(range(num_rollouts), desc="Rollouts"):
            rng = jax.random.PRNGKey(rollout_idx)
            result = video_rollout(
                tokenizer=tokenizer,
                dynamics=dynamics,
                policy=actions_future,  # Use ground truth actions from dataset
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
            print(f"    Output frames shape: {pred_frames.shape}")
            print(f"    Output latents shape: {pred_latents.shape}")

            # Verify shapes
            assert pred_frames.shape[0] == B, f"Batch size mismatch"
            assert (
                pred_frames.shape[1] == T_ctx + horizon
            ), f"Time steps mismatch: {pred_frames.shape[1]} vs {T_ctx + horizon}"
            assert (
                pred_frames.shape[2] == H and pred_frames.shape[3] == W
            ), f"Frame size mismatch"
            assert pred_frames.shape[4] == C, f"Channel mismatch"
            print(f"    ✓ Shape checks passed")

            # Save as GIF with 2 fps
            try:
                from PIL import Image

                sample_idx = 0
                frames_list = []
                for t in range(T_ctx + horizon):
                    frame = np.array(pred_frames[sample_idx, t])  # Convert JAX to numpy
                    frame_np = np.clip(frame, 0, 255).astype(np.uint8)
                    frames_list.append(Image.fromarray(frame_np))

                save_path = output_path / f"rollout_{rollout_idx}.gif"
                frames_list[0].save(
                    save_path,
                    save_all=True,
                    append_images=frames_list[1:],
                    duration=500,  # 500ms per frame = 2 fps
                    loop=0,
                )
                print(f"    ✓ Saved GIF to {save_path}")
            except ImportError:
                print(f"    (PIL not available, skipping visualization)")

        print("\n✓ video_rollout tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test generation.py with a checkpoint")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
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

    print(f"\n{'='*60}")
    print("Generation.py Test")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Horizon: {args.horizon}")
    print(f"Output directory: {args.output_dir}")

    try:
        test_video_rollout(
            str(checkpoint_dir),
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
