"""
Standalone evaluation script for dynamics models.

Loads a pretrained dynamics model checkpoint and runs evaluation
(video prediction visualization and metrics computation).

Usage:
    python scripts/eval_dynamics.py \
        --checkpoint_path /path/to/dynamics/checkpoint \
        --output_dir /path/to/output \
        --num_samples 4 \
        --horizon 60
"""
import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.data import make_iterator
from dreamer.configs import DatasetConfig, DataloaderConfig
from dreamer.parallel import build_parallel
from dreamer.training import run_evaluation

logging.getLogger('absl').setLevel(logging.WARNING)

jax.config.update("jax_compilation_cache_dir", "/scratch/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


class DummyLogger:
    """Minimal logger for evaluation that just prints to stdout."""
    
    def log(self, step, **kwargs):
        pass
    
    def log_video(self, step, key, video, fps=5):
        pass
    
    def log_metrics(self, step, metrics, prefix=None):
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamics model checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to dynamics checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: checkpoint_path/eval)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to validation data (ArrayRecord format). If not provided, uses checkpoint config.",
    )
    parser.add_argument(
        "--index_max",
        type=int,
        default=None,
        help="Number of data shards to use (required for latent datasets)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of validation samples to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=16,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        default="data",
        choices=["data", "fsdp", "tensor_parallel"],
        help="Parallelism strategy",
    )
    parser.add_argument(
        "--omega",
        type=float,
        nargs='+',
        default=[0.0],
        help="Guidance strength(s) (0 = no guidance). Can specify multiple values. See https://arxiv.org/pdf/2502.07849",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Guidance alpha parameter for scaling",
    )
    args = parser.parse_args()

    # Setup output directory
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if args.output_dir is None:
        output_dir = checkpoint_path / "eval"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Saving visualizations to: {output_dir}")

    # Build parallel context
    mesh, data_sharding, mesh_rules = build_parallel(args.parallel_strategy)

    with jax.set_mesh(mesh):
        rng = jax.random.PRNGKey(args.seed)

        # Load pretrained dynamics bundle (includes tokenizer)
        print("Loading dynamics model...")
        bundle = DynamicsCheckpointBundle.from_pretrained(
            str(checkpoint_path),
            mesh_rules=mesh_rules,
            rngs=nnx.Rngs(0),
        )
        tokenizer = bundle.tokenizer
        dynamics = bundle.dynamics
        
        print(f"Loaded dynamics model with config: {dynamics.cfg}")

        # Determine data type from dynamics config
        use_latent_data = hasattr(dynamics.cfg, 'latent_mean') and dynamics.cfg.latent_mean is not None

        # Build dataset config for validation data
        if args.data_path is not None:
            data_path = args.data_path
        else:
            # Try to infer from checkpoint or use default
            data_path = "/datasets/minecraft_vpt_latent"
            print(f"No data path provided, using default: {data_path}")

        # Determine index_max for shard-based datasets
        index_max = args.index_max
        if index_max is None and use_latent_data:
            # Try to auto-detect number of shards
            data_dir = Path(data_path)
            if data_dir.exists():
                shards = list(data_dir.glob("shard-*.array_record"))
                index_max = len(shards)
            if index_max is None or index_max == 0:
                raise ValueError(
                    f"Could not auto-detect number of shards in {data_path}. "
                    f"Please specify --index_max or ensure the data directory contains shard-*.array_record files."
                )
            print(f"Auto-detected {index_max} shards in {data_path}")

        # Create minimal dataset config for evaluation
        dataloader_cfg = DataloaderConfig(
            B=args.batch_size,
            T=args.seq_length,
            num_workers=2,
            prefetch_buffer_size=2,
            device_prefetch_buffer_size=1,
        )
        
        dataset_cfg = DatasetConfig(
            name="minecraft_vpt",
            data_type="latent" if use_latent_data else "video",
            array_record_path=data_path,
            dataloader_cfg=dataloader_cfg,
            index_max=index_max if index_max else 0,
            # Use defaults for dataset stats - these should match training
            dataset_std=(1.0, 1.0, 1.0),
        )

        # Create data iterator
        print("Loading validation data...")
        dataloader = make_iterator(dataset_cfg, seed=args.seed, device=data_sharding)
        
        # Get a batch for evaluation
        batch = next(iter(dataloader))
        
        if use_latent_data:
            val_data = batch.get("latents", batch.get("videos"))
        else:
            val_data = batch["videos"]
        val_actions = batch["actions"]
        
        # Limit to num_samples
        val_data = val_data[:args.num_samples]
        val_actions = val_actions[:args.num_samples]

        print(f"Validation data shape: {val_data.shape}")
        print(f"Validation actions: binary={val_actions.binary.shape if val_actions.binary is not None else None}, "
              f"categorical={val_actions.categorical.shape if val_actions.categorical is not None else None}")

        # Create a minimal config-like object for run_evaluation
        class EvalConfig:
            def __init__(self, dataset_std):
                self.dataset = type('obj', (object,), {'dataset_std': dataset_std})()
        
        eval_cfg = EvalConfig(dataset_std=dataset_cfg.dataset_std)

        # Run evaluation for each omega value
        print("Running evaluation...")
        logger = DummyLogger()
        
        omega_values = args.omega
        print(f"Evaluating with omega values: {omega_values}")
        
        for omega in omega_values:
            omega_str = f"omega_{omega:.2f}".replace("-", "neg").replace(".", "p")
            omega_dir = output_dir / omega_str
            omega_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n--- Evaluating with omega={omega}, alpha={args.alpha} ---")
            
            run_evaluation(
                cfg=eval_cfg,
                step=0,
                tokenizer=tokenizer,
                dynamics=dynamics,
                val_data=val_data,
                val_actions=val_actions,
                use_latent_data=use_latent_data,
                vis_dir=omega_dir,
                rng=rng,
                logger=logger,
                omega=jnp.array(omega),
                alpha=jnp.array(args.alpha),
            )

        print(f"\nEvaluation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
