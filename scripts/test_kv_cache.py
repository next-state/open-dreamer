#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.configs import DatasetConfig
from dreamer.data import make_iterator
from dreamer.parallel import build_parallel
from dreamer.utils import from_dict, normalize_latents


def load_dataset_cfg(path: str) -> DatasetConfig:
    cfg_raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if isinstance(cfg_raw, dict):
        cfg_raw.pop("defaults", None)
    return from_dict(DatasetConfig, cfg_raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache equivalence test (full vs cached last frame).")
    parser.add_argument("--ckpt", required=True, help="Path to dynamics checkpoint directory.")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML (e.g. configs/dataset/minecraft_vpt_latent.yaml).")
    parser.add_argument("--parallel", default="data", choices=["data", "fsdp", "tp", "sp"], help="Parallel strategy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    dataset_cfg = load_dataset_cfg(args.dataset)

    # Build parallel mesh
    mesh, _, mesh_rules = build_parallel(args.parallel)

    with jax.set_mesh(mesh):
        # Load dynamics + tokenizer
        bundle = DynamicsCheckpointBundle.from_pretrained(args.ckpt, mesh_rules=mesh_rules, rngs=nnx.Rngs(0))
        dynamics = bundle.dynamics
        tokenizer = bundle.tokenizer

        if dynamics.cfg.use_seq_parallel:
            raise RuntimeError("KV cache test expects use_seq_parallel=False (cache path does not use SP all_gather).")

        # Sample a batch
        it = make_iterator(dataset_cfg, seed=args.seed)
        batch = next(it)

        if dataset_cfg.data_type == "latent":
            latents = batch["latents"]
        else:
            videos = batch["videos"]
            latents, _ = tokenizer.encode(videos, deterministic=True)
            latents = jax.lax.stop_gradient(latents)

        actions = batch["actions"]

        # Prepare inputs
        latents = latents.astype(dynamics.dtype)
        latents = normalize_latents(latents, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

        B, T, n_latents, d_latent = latents.shape
        if T < 2:
            raise ValueError(f"Need T>=2 for cache test, got T={T}")

        rng = jax.random.PRNGKey(args.seed)
        rng, noise_key = jax.random.split(rng)

        # Last frame fully noise
        z_in = latents
        z_noise = jax.random.normal(noise_key, (B, 1, n_latents, d_latent), dtype=latents.dtype)
        z_in = z_in.at[:, -1:].set(z_noise)

        k_max = dynamics.cfg.k_max
        emax = int(math.log2(k_max))
        step_idx_full = jnp.full((B, T), emax, dtype=jnp.int32)
        tau_idx_full = jnp.full((B, T), k_max - 1, dtype=jnp.int32)
        tau_idx_full = tau_idx_full.at[:, -1:].set(0)  # fully noise on last frame

        # Full forward (no cache)
        full_out, _ = dynamics(
            actions, step_idx_full, tau_idx_full, z_in,
            deterministic=True, caches=None
        )
        full_last = full_out[:, -1]

        # Cache prefill with context frames
        caches = dynamics.create_static_caches(
            batch_size=B,
            n_latents=n_latents,
            window_size=T,
            n_agent=0,
            dtype=latents.dtype,
        )
        actions_ctx = actions[:, :-1]
        step_idx_ctx = step_idx_full[:, :-1]
        tau_idx_ctx = tau_idx_full[:, :-1]
        z_ctx = z_in[:, :-1]

        _, (_, caches_prefilled) = dynamics(
            actions_ctx, step_idx_ctx, tau_idx_ctx, z_ctx,
            deterministic=True, caches=caches
        )

        # Cached forward for last frame only
        actions_last = actions[:, -1:]
        step_idx_last = step_idx_full[:, -1:]
        tau_idx_last = tau_idx_full[:, -1:]
        z_last = z_in[:, -1:]

        cached_out, _ = dynamics(
            actions_last, step_idx_last, tau_idx_last, z_last,
            deterministic=True, caches=caches_prefilled
        )
        cached_last = cached_out[:, -1]

        # Compare
        diff = jnp.abs(full_last - cached_last)
        max_diff = float(jnp.max(diff))
        mean_diff = float(jnp.mean(diff))

        print(f"KV cache test: max_abs_diff={max_diff:.6e}  mean_abs_diff={mean_diff:.6e}")


if __name__ == "__main__":
    main()
