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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_dir}...")
    mesh, _, mesh_rules = build_parallel("data")

    with jax.set_mesh(mesh):
        bundle = DynamicsCheckpointBundle.from_pretrained(
            checkpoint_dir, mesh_rules=mesh_rules
        )
        dynamics = bundle.dynamics
        tokenizer = bundle.tokenizer

        H = tokenizer.cfg.decoder.H
        W = tokenizer.cfg.decoder.W
        C = 3
        T_ctx = 4

        print(f"Frame dimensions: {H}x{W}x{C}")
        print(f"Context length: {T_ctx}, Horizon: {horizon}")

        k_max = dynamics.cfg.k_max
        schedule = DenoiseSchedule.init(n_steps=4, k_max=k_max)

        # JIT-compiled decoder (defined once, outside loop)
        @nnx.jit
        def decode_jit(z):
            frames, _ = tokenizer.decode(z, deterministic=True)
            return frames

        # Load real data — use num_rollouts as batch size
        print(f"Loading dataset (data_type={dataset_cfg.data_type})...")
        dataset_cfg.B = num_rollouts
        dataset_cfg.T = T_ctx + horizon + 4  # Extra buffer

        short_loader, _ = make_dual_iterators(
            dataset_cfg,
            short_T=dataset_cfg.T,
            long_T=dataset_cfg.T,
            num_workers=0,
        )

        batch = next(iter(short_loader))
        actions = batch["actions"]

        if use_latent_data:
            data = batch["latents"]
        else:
            data = batch["videos"]

        B = data.shape[0]
        data_ctx = data[:, :T_ctx]
        actions_ctx = actions[:, :T_ctx]
        actions_future = actions[:, T_ctx : T_ctx + horizon]

        print(f"Loaded batch: data {data.shape}, B={B}")
        print(f"Context data shape: {data_ctx.shape}")

        # Decode ground truth for comparison
        if use_latent_data:
            gt_latents = data[:, :T_ctx + horizon]
            gt_frames = decode_jit(gt_latents.astype(jnp.bfloat16))
            gt_frames = jnp.clip(gt_frames, 0, 255).astype(jnp.uint8)
        else:
            gt_frames = data[:, :T_ctx + horizon]

        # Run rollout (single batched call)
        print(f"\nRunning rollout (B={B}, horizon={horizon})...")
        rng = jax.random.PRNGKey(0)

        if use_latent_data:
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
            pred_frames = decode_jit(result["latents"])
            pred_frames = jnp.clip(pred_frames, 0, 255).astype(jnp.uint8)
            pred_latents = result["latents"]
        else:
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
            pred_frames = jnp.clip(result["frames"], 0, 255).astype(jnp.uint8)
            pred_latents = result["latents"]

        print(f"Output frames shape: {pred_frames.shape}")
        print(f"Output latents shape: {pred_latents.shape}")

        # Verify shapes
        assert pred_frames.shape == (B, T_ctx + horizon, H, W, C), \
            f"Shape mismatch: {pred_frames.shape} vs {(B, T_ctx + horizon, H, W, C)}"
        print("✓ Shape checks passed")

        # Save GIFs: ground truth and prediction side by side per batch element
        try:
            from PIL import Image

            for b in range(B):
                frames_list = []
                for t in range(T_ctx + horizon):
                    gt_frame = np.array(gt_frames[b, t]).clip(0, 255).astype(np.uint8)
                    pred_frame = np.array(pred_frames[b, t]).clip(0, 255).astype(np.uint8)
                    # Stack GT (top) and pred (bottom)
                    combined = np.concatenate([gt_frame, pred_frame], axis=0)
                    frames_list.append(Image.fromarray(combined))

                save_path = output_path / f"rollout_{b}.gif"
                frames_list[0].save(
                    save_path,
                    save_all=True,
                    append_images=frames_list[1:],
                    duration=500,  # 2 fps
                    loop=0,
                )
                print(f"  ✓ Saved GIF to {save_path} (top=GT, bottom=pred)")
        except ImportError:
            print("  (PIL not available, skipping visualization)")

        print("\n✓ rollout tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test generation.py with a checkpoint")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
    parser.add_argument(
        "--dataset_cfg", default="configs/dataset/minecraft_vpt_latent.yaml", help="Path to dataset config (YAML)."
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=2, help="Number of rollouts (= batch size)"
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

    import yaml
    from dreamer.configs import DatasetConfig

    cfg_path = Path(args.dataset_cfg)
    if not cfg_path.exists():
        print(f"Error: dataset config file {cfg_path} not found")
        return 1

    with open(cfg_path) as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict.pop("defaults", None)
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
