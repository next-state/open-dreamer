#!/usr/bin/env python3
"""
Test script for generation.py: loads a checkpoint and runs rollouts on real data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python test_generation.py --checkpoint_dir /path/to/checkpoint [--num_rollouts 2] [--horizon 4]
"""

import argparse
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.generation import DenoiseSchedule, latent_rollout, video_rollout
from dreamer.parallel import build_parallel
from dreamer.data import make_dual_iterators


def test_rollout(
    checkpoint_dir: str,
    dataset_cfg: "DatasetConfig",
    num_rollouts: int = 2,
    horizon: int = 4,
    output_dir: str = "test_outputs",
):
    """Test rollout with real data from dataset."""
    use_latent_data = dataset_cfg.data_type == "latent"

    print("\n" + "=" * 60)
    print(f"Testing {'latent' if use_latent_data else 'video'}_rollout")
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
        C = 3
        T_ctx = 4

        print(f"Frame dimensions: {H}x{W}x{C}")
        print(f"Context length: {T_ctx}, Horizon: {horizon}")

        # Get dynamics config
        k_max = dynamics.cfg.k_max
        schedule = DenoiseSchedule.init(n_steps=4, k_max=k_max)

        # Load real data from dataset
        print(f"Loading dataset (data_type={dataset_cfg.data_type})...")
        short_loader, _ = make_dual_iterators(
            dataset_cfg,
            short_T=T_ctx + horizon + 4,  # Extra buffer
            long_T=T_ctx + horizon + 4,
            num_workers=0,
        )

        iterator = iter(short_loader)
        batch = next(iterator)
        actions = batch["actions"]

        if use_latent_data:
            data = batch["latents"]  # (B, T, n_latents, d_bottleneck)
        else:
            data = batch["videos"]  # (B, T, H, W, C)

        B, T = data.shape[0], data.shape[1]
        T_total = min(T_ctx + horizon, T)

        data_all = data[:, :T_total]
        actions_all = actions[:, :T_total]
        data_ctx = data_all[:, :T_ctx]
        actions_ctx = actions_all[:, :T_ctx]
        actions_future = actions_all[:, T_ctx : T_ctx + horizon]

        print(f"Loaded batch: data {data.shape}, B={B}")
        print(f"Context data shape: {data_ctx.shape}")
        if actions_ctx.binary is not None:
            print(f"Context actions: binary {actions_ctx.binary.shape}, categorical {actions_ctx.categorical.shape}")

        # Test rollout
        print(f"\nRunning rollout...")
        for rollout_idx in tqdm(range(num_rollouts), desc="Rollouts"):
            rng = jax.random.PRNGKey(rollout_idx)

            if use_latent_data:
                # Latent path: use latent_rollout directly, then decode
                latents_ctx = data_ctx.astype(jnp.bfloat16)
                result = latent_rollout(
                    dynamics=dynamics,
                    policy=actions_future,
                    schedule=schedule,
                    latents_ctx=latents_ctx,
                    actions_ctx=actions_ctx,
                    num_steps=horizon,
                    rng=rng,
                    initial_task_embedding=None,
                    deterministic=True,
                )

                # Decode latents to frames for visualization
                @nnx.jit
                def decode_jit(z):
                    frames, _ = tokenizer.decode(z, deterministic=True)
                    return frames

                pred_frames = decode_jit(result["latents"])
                pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
                pred_latents = result["latents"]
            else:
                # Video path: use video_rollout
                result = video_rollout(
                    tokenizer=tokenizer,
                    dynamics=dynamics,
                    policy=actions_future,
                    schedule=schedule,
                    frames_ctx=data_ctx,
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
            assert pred_frames.shape[0] == B
            assert pred_frames.shape[1] == T_ctx + horizon
            assert pred_frames.shape[2] == H and pred_frames.shape[3] == W
            assert pred_frames.shape[4] == C
            print(f"    ✓ Shape checks passed")

            # Save as GIF with 2 fps
            try:
                from PIL import Image

                sample_idx = 0
                frames_list = []
                for t in range(T_ctx + horizon):
                    frame = np.array(pred_frames[sample_idx, t])
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

        print("\n✓ rollout tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test generation.py with a checkpoint")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--dataset_cfg", default="configs/dataset/minecraft_vpt_latent.yaml", help="Path to dataset config (YAML)."
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
    import yaml
    from dreamer.configs import DatasetConfig

    cfg_path = Path(args.dataset_cfg)
    if not cfg_path.exists():
        print(f"Error: dataset config file {cfg_path} not found")
        return 1

    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Config YAML may have flat keys or nested under 'dataset:'
    cfg_dict.pop("defaults", None)  # Remove hydra defaults key
    dataset_cfg = DatasetConfig(**cfg_dict)

    print(f"\n{'='*60}")
    print("Generation.py Test")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Dataset config: {args.dataset_cfg} (data_type={dataset_cfg.data_type})")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Horizon: {args.horizon}")
    print(f"Output directory: {args.output_dir}")

    try:
        test_rollout(
            str(checkpoint_dir),
            dataset_cfg=dataset_cfg,
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
