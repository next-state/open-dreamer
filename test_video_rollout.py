#!/usr/bin/env python3
"""
Testing script for video rollout with history guidance.

This script:
1. Loads a pre-trained dynamics model checkpoint
2. Loads a batch of data
3. Takes the first 4 frames as context
4. Performs a video rollout with default history guidance
"""

import jax
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
import argparse
from einops import rearrange
import imageio.v3 as iio

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.configs import HistoryGuidanceConfig, DatasetConfig
from dreamer.generation import DenoiseSchedule
from dreamer.sampler import sample_video
from dreamer.parallel import build_parallel
from dreamer.data import make_iterator
from dreamer.utils import apply_border


def test_video_rollout(
    checkpoint_path: str,
    num_generation_steps: int = 16,
    batch_size: int = 2,
    context_frames: int = 4,
    use_guidance: bool = True,
    seed: int = 42,
    output_video_path: str | None = None,
):
    """
    Test video rollout with history guidance.

    Args:
        checkpoint_path: Path to the dynamics checkpoint
        num_generation_steps: Number of frames to generate
        batch_size: Batch size for generation
        context_frames: Number of context frames
        use_guidance: Whether to use history guidance
        seed: Random seed
        output_video_path: Path to save output video (optional)
    """

    print("=" * 80)
    print("VIDEO ROLLOUT TEST WITH HISTORY GUIDANCE")
    print("=" * 80)

    # Setup RNG
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Load checkpoint
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)

    # Setup mesh and partitioning rules for single-device (CPU/GPU)
    print("   Setting up JAX mesh context...")
    mesh, data_sharding, mesh_rules = build_parallel("data")

    # Enter mesh context before loading models
    with jax.set_mesh(mesh):
        # Load the dynamics checkpoint bundle
        bundle = DynamicsCheckpointBundle.from_pretrained(
            str(checkpoint_path),
            mesh_rules=mesh_rules,
            rngs=nnx.Rngs(init_rng),
        )

        tokenizer = bundle.tokenizer
        dynamics = bundle.dynamics

        print(f"   ✓ Loaded dynamics model")
        print(f"   ✓ Loaded tokenizer")
        print(f"   Model config: latent_mean={dynamics.cfg.latent_mean}, latent_std={dynamics.cfg.latent_std}")

        # Create denoising schedule
        print(f"\n2. Creating denoising schedule")
        schedule = DenoiseSchedule.init(num_steps=4, k_max=dynamics.cfg.k_max)
        print(f"   ✓ Schedule created with {schedule.num_steps} steps")

        # Load real data
        print(f"\n3. Loading data")
        dataset_config = DatasetConfig(
            name="minecraft_vpt_latent",
            data_type="latent",
            array_record_path="/scratch/tokenized_dataset",
            index_max=25,
            B=batch_size,
            T=context_frames + num_generation_steps,
            num_binary_actions=23,
            categorical_action_dim=121,
            continuous_action_dim=0,
        )

        # Create data iterator
        dataloader = make_iterator(
            dataset_config,
            num_workers=4,
            prefetch_buffer_size=1,
            seed=42,
            print_filter_warnings=False,
        )
        iterator = iter(dataloader)

        # Load one batch
        batch = next(iterator)
        print(f"   ✓ Loaded batch from dataset")

        # Extract latents and actions
        latents_full = jax.device_put(batch["latents"], data_sharding)
        actions_full = jax.device_put(batch["actions"], data_sharding)


        # Setup history guidance
        print(f"\n4. Setting up history guidance")
        if use_guidance:
            guidance_config = HistoryGuidanceConfig(
                enabled=True,
                guidance_type="tf",
                omega=1.5,
                tau_H_frac=0.5,
                history_long=16,
                history_short=4,
                omega_long=0.5,
                omega_short=1.0,
                omega_frac=0.5,
            )
            print(f"   ✓ History guidance enabled (type={guidance_config.guidance_type})")
            print(f"     - omega: {guidance_config.omega}")
            print(f"     - tau_H_frac: {guidance_config.tau_H_frac}")
        else:
            guidance_config = None
            print(f"   ✓ History guidance disabled")

        # Perform video rollout with sample_video
        print(f"\n5. Running video rollout with sample_video")
        print(f"   Generating {num_generation_steps} frames from {context_frames} context frames...")

        rng, rollout_rng = jax.random.split(rng)

        try:
            # Use sample_video with un-decoded latents
            pred_frames, gt_decoded_frames, _ = sample_video(
                tokenizer=tokenizer,
                dynamics=dynamics,
                frames=None,  # Use latents instead
                actions=actions_full,
                horizon=num_generation_steps,
                schedule_config=schedule,
                rng=rollout_rng,
                policy=None,  # Use ground truth actions
                task_embedder=None,
                latents=latents_full,
                guidance_config=guidance_config,
            )

            print(f"   ✓ Rollout completed successfully!")
            print(f"   ✓ Pred frames shape: {pred_frames.shape}")
            print(f"   ✓ GT decoded frames shape: {gt_decoded_frames.shape}")
            print(f"     - Total: {pred_frames.shape[1]} frames ({context_frames} context + {num_generation_steps} generated)")
            print(f"     - Resolution: {pred_frames.shape[2]}x{pred_frames.shape[3]}")

            # Verify output properties
            print(f"\n6. Output verification")
            assert pred_frames.dtype == jnp.uint8, f"Expected uint8, got {pred_frames.dtype}"
            print(f"   ✓ Output dtype is uint8")

            assert pred_frames.shape[0] == batch_size, f"Batch size mismatch"
            print(f"   ✓ Batch size correct: {pred_frames.shape[0]}")

            expected_total = context_frames + num_generation_steps
            assert pred_frames.shape[1] == expected_total, f"Total frames mismatch: expected {expected_total}, got {pred_frames.shape[1]}"
            print(f"   ✓ Total frames correct: {pred_frames.shape[1]} ({context_frames} context + {num_generation_steps} generated)")

            # Apply border to context frames in predictions
            print(f"\n7. Processing frames for video output")
            pred_frames = pred_frames.at[:, :context_frames].set(
                apply_border(pred_frames[:, :context_frames])
            )
            print(f"   ✓ Applied red border to context frames")

            # Stack GT and predicted frames
            frames_list = [gt_decoded_frames, pred_frames]
            stacked_frames = jnp.stack(frames_list)

            # Take only the first batch_size samples
            num_videos = batch_size
            stacked_frames = stacked_frames[:, :num_videos]
            print(f"   ✓ Stacked frames shape: {stacked_frames.shape}")

            # Rearrange for video format: (S, B, T, H, W, C) -> (T, B*H, S*W, C)
            videos = rearrange(stacked_frames, 'S B T H W C -> T (B H) (S W) C', B=num_videos)
            print(f"   ✓ Rearranged video shape: {videos.shape}")

            # Save video if output path provided
            if output_video_path:
                print(f"\n8. Saving video")
                output_path = Path(output_video_path) / 'guidance.mp4'
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert to numpy and ensure uint8
                video_np = jnp.asarray(videos, dtype=jnp.uint8)

                # Save as MP4
                iio.imwrite(str(output_path), video_np, fps=5, plugin='pyav', codec='libx264')
                print(f"   ✓ Saved video to: {output_path.resolve()}")

            print(f"\n{'='*80}")
            print(f"✓ TEST PASSED")
            print(f"{'='*80}\n")

            return videos

        except Exception as e:
            print(f"\n✗ TEST FAILED")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test video rollout with history guidance"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to dynamics checkpoint",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=252,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=4,
        help="Number of context frames",
    )
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="Disable history guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output video (e.g., video.mp4)",
    )

    args = parser.parse_args()

    test_video_rollout(
        checkpoint_path=args.checkpoint,
        num_generation_steps=args.num_steps,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        use_guidance=not args.no_guidance,
        seed=args.seed,
        output_video_path=args.output,
    )
