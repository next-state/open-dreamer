#!/usr/bin/env python3
import argparse
import math
import sys

import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from dreamer.checkpointing import DynamicsCheckpointBundle
from dreamer.configs import DatasetConfig
from dreamer.data import make_iterator
from dreamer.parallel import build_parallel
from dreamer.utils import from_dict, normalize_latents, build_dart_time_mask


def load_dataset_cfg(path: str) -> DatasetConfig:
    cfg_raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if isinstance(cfg_raw, dict):
        cfg_raw.pop("defaults", None)
    return from_dict(DatasetConfig, cfg_raw)


def prepare_batch(bundle, dataset_cfg, seed: int):
    it = make_iterator(dataset_cfg, seed=seed)
    batch = next(it)

    if dataset_cfg.data_type == "latent":
        latents = batch["latents"]
    else:
        videos = batch["videos"]
        latents, _ = bundle.tokenizer.encode(videos, deterministic=True)
        latents = jax.lax.stop_gradient(latents)

    actions = batch["actions"]

    latents = latents.astype(bundle.dynamics.dtype)
    latents = normalize_latents(latents, bundle.dynamics.cfg.latent_mean, bundle.dynamics.cfg.latent_std)

    B, T, n_latents, d_latent = latents.shape
    if T < 3:
        raise ValueError(f"Need T>=3 for consistency tests, got T={T}")

    rng = jax.random.PRNGKey(seed)
    rng, noise_key = jax.random.split(rng)
    z_noise = jax.random.normal(noise_key, (B, 1, n_latents, d_latent), dtype=latents.dtype)
    latents = latents.at[:, -1:].set(z_noise)  # last frame fully noise

    k_max = bundle.dynamics.cfg.k_max
    emax = int(math.log2(k_max))
    step_idx = jnp.full((B, T), emax, dtype=jnp.int32)
    tau_idx = jnp.full((B, T), k_max - 1, dtype=jnp.int32)
    tau_idx = tau_idx.at[:, -1:].set(0)  # fully noise

    return latents, actions, step_idx, tau_idx


def test_cached_vs_full_last(bundle, latents, actions, step_idx, tau_idx, tol):
    B, T, n_latents, _ = latents.shape

    full_out, _ = bundle.dynamics(
        actions, step_idx, tau_idx, latents,
        deterministic=True, caches=None
    )
    full_last = full_out[:, -1]

    caches = bundle.dynamics.create_static_caches(
        batch_size=B,
        n_latents=n_latents,
        window_size=T,
        n_agent=0,
        dtype=latents.dtype,
    )
    _, (_, caches_prefilled) = bundle.dynamics(
        actions[:, :-1],
        step_idx[:, :-1],
        tau_idx[:, :-1],
        latents[:, :-1],
        deterministic=True,
        caches=caches,
    )
    cached_out, _ = bundle.dynamics(
        actions[:, -1:],
        step_idx[:, -1:],
        tau_idx[:, -1:],
        latents[:, -1:],
        deterministic=True,
        caches=caches_prefilled,
    )
    cached_last = cached_out[:, -1]

    diff = jnp.abs(full_last.astype(jnp.float32) - cached_last.astype(jnp.float32))
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert max_diff <= tol, f"cached_vs_full_last max_diff {max_diff:.3e} > {tol:.3e} (mean {mean_diff:.3e})"


def test_cached_vs_full_multistep(bundle, latents, actions, step_idx, tau_idx, tol):
    B, T, n_latents, _ = latents.shape

    full_out, _ = bundle.dynamics(
        actions, step_idx, tau_idx, latents,
        deterministic=True, caches=None
    )
    full_tail = full_out[:, -2:]

    caches = bundle.dynamics.create_static_caches(
        batch_size=B,
        n_latents=n_latents,
        window_size=T,
        n_agent=0,
        dtype=latents.dtype,
    )
    _, (_, caches_prefilled) = bundle.dynamics(
        actions[:, :-2],
        step_idx[:, :-2],
        tau_idx[:, :-2],
        latents[:, :-2],
        deterministic=True,
        caches=caches,
    )
    out1, (_, caches_mid) = bundle.dynamics(
        actions[:, -2:-1],
        step_idx[:, -2:-1],
        tau_idx[:, -2:-1],
        latents[:, -2:-1],
        deterministic=True,
        caches=caches_prefilled,
    )
    out2, _ = bundle.dynamics(
        actions[:, -1:],
        step_idx[:, -1:],
        tau_idx[:, -1:],
        latents[:, -1:],
        deterministic=True,
        caches=caches_mid,
    )
    cached_tail = jnp.concatenate([out1, out2], axis=1)

    diff = jnp.abs(full_tail.astype(jnp.float32) - cached_tail.astype(jnp.float32))
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert max_diff <= tol, f"cached_vs_full_multistep max_diff {max_diff:.3e} > {tol:.3e} (mean {mean_diff:.3e})"


def test_causal_truncation(bundle, latents, actions, step_idx, tau_idx, tol):
    B, T, *_ = latents.shape
    cut = max(2, T // 2)

    full_out, _ = bundle.dynamics(
        actions, step_idx, tau_idx, latents,
        deterministic=True, caches=None
    )
    trunc_out, _ = bundle.dynamics(
        actions[:, :cut],
        step_idx[:, :cut],
        tau_idx[:, :cut],
        latents[:, :cut],
        deterministic=True,
        caches=None
    )

    diff = jnp.abs(full_out[:, :cut].astype(jnp.float32) - trunc_out.astype(jnp.float32))
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert max_diff <= tol, f"causal_truncation max_diff {max_diff:.3e} > {tol:.3e} (mean {mean_diff:.3e})"


def _forward_dart(bundle, latents_clean, latents_noisy, actions, step_idx_noisy, tau_idx_noisy):
    B, T, n_latents, _ = latents_clean.shape
    k_max = bundle.dynamics.cfg.k_max
    emax = int(math.log2(k_max))

    step_idx_clean = jnp.full((B, T), emax, dtype=jnp.int32)
    tau_idx_clean = jnp.full((B, T), k_max - 1, dtype=jnp.int32)

    step_idx_full = jnp.concatenate([step_idx_clean, step_idx_noisy], axis=1)
    tau_idx_full = jnp.concatenate([tau_idx_clean, tau_idx_noisy], axis=1)
    latents_full = jnp.concatenate([latents_clean, latents_noisy], axis=1)
    actions_full = jax.tree.map(lambda x: jnp.concatenate([x, x], axis=1) if x is not None else None, actions)

    time_mask = build_dart_time_mask(T)

    out, _ = bundle.dynamics(
        actions_full, step_idx_full, tau_idx_full, latents_full,
        time_mask=time_mask, use_dart=True, deterministic=True, caches=None
    )
    return out


def test_dart_clean_independent_from_noisy(bundle, latents, actions, tol, seed):
    B, T, n_latents, d_latent = latents.shape
    rng = jax.random.PRNGKey(seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    lat_noisy_a = jax.random.normal(k1, (B, T, n_latents, d_latent), dtype=latents.dtype)
    lat_noisy_b = jax.random.normal(k2, (B, T, n_latents, d_latent), dtype=latents.dtype)

    step_idx_noisy = jnp.full((B, T), int(math.log2(bundle.dynamics.cfg.k_max)), dtype=jnp.int32)
    tau_idx_noisy = jnp.zeros((B, T), dtype=jnp.int32)

    out_a = _forward_dart(bundle, latents, lat_noisy_a, actions, step_idx_noisy, tau_idx_noisy)
    out_b = _forward_dart(bundle, latents, lat_noisy_b, actions, step_idx_noisy, tau_idx_noisy)

    diff = jnp.abs(out_a[:, :T].astype(jnp.float32) - out_b[:, :T].astype(jnp.float32))
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert max_diff <= tol, f"dart_clean_independent max_diff {max_diff:.3e} > {tol:.3e} (mean {mean_diff:.3e})"


def test_dart_noisy_not_same_time_clean(bundle, latents, actions, tol, seed):
    B, T, n_latents, d_latent = latents.shape
    t = max(1, T // 2)
    rng = jax.random.PRNGKey(seed)
    rng, k1, k2 = jax.random.split(rng, 3)

    lat_noisy = jax.random.normal(k1, (B, T, n_latents, d_latent), dtype=latents.dtype)
    lat_clean_mod = latents.at[:, t].set(latents[:, t] + 0.1 * jax.random.normal(k2, latents[:, t].shape, dtype=latents.dtype))

    step_idx_noisy = jnp.full((B, T), int(math.log2(bundle.dynamics.cfg.k_max)), dtype=jnp.int32)
    tau_idx_noisy = jnp.zeros((B, T), dtype=jnp.int32)

    out_base = _forward_dart(bundle, latents, lat_noisy, actions, step_idx_noisy, tau_idx_noisy)
    out_mod = _forward_dart(bundle, lat_clean_mod, lat_noisy, actions, step_idx_noisy, tau_idx_noisy)

    diff = jnp.abs(out_base[:, T + t].astype(jnp.float32) - out_mod[:, T + t].astype(jnp.float32))
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert max_diff <= tol, f"dart_noisy_not_same_time_clean max_diff {max_diff:.3e} > {tol:.3e} (mean {mean_diff:.3e})"


def test_dart_noisy_not_other_noisy(bundle, latents, actions, tol, seed):
    B, T, n_latents, d_latent = latents.shape
    t = max(1, T // 2)
    t2 = 0 if t != 0 else 1
    rng = jax.random.PRNGKey(seed)
    rng, k1, k2 = jax.random.split(rng, 3)

    lat_noisy = jax.random.normal(k1, (B, T, n_latents, d_latent), dtype=latents.dtype)
    lat_noisy_mod = lat_noisy.at[:, t2].set(lat_noisy[:, t2] + 0.1 * jax.random.normal(k2, lat_noisy[:, t2].shape, dtype=latents.dtype))

    step_idx_noisy = jnp.full((B, T), int(math.log2(bundle.dynamics.cfg.k_max)), dtype=jnp.int32)
    tau_idx_noisy = jnp.zeros((B, T), dtype=jnp.int32)

    out_base = _forward_dart(bundle, latents, lat_noisy, actions, step_idx_noisy, tau_idx_noisy)
    out_mod = _forward_dart(bundle, latents, lat_noisy_mod, actions, step_idx_noisy, tau_idx_noisy)

    diff = jnp.abs(out_base[:, T + t].astype(jnp.float32) - out_mod[:, T + t].astype(jnp.float32))
    max_diff = float(jnp.max(diff))
    mean_diff = float(jnp.mean(diff))
    assert max_diff <= tol, f"dart_noisy_not_other_noisy max_diff {max_diff:.3e} > {tol:.3e} (mean {mean_diff:.3e})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Consistency tests for Dreamer JAX models.")
    parser.add_argument("--ckpt", required=True, help="Path to dynamics checkpoint directory.")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML.")
    parser.add_argument("--parallel", default="data", choices=["data", "fsdp", "tp", "sp"], help="Parallel strategy.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--tol", type=float, default=3e-4, help="Max abs diff tolerance.")
    parser.add_argument("--no_dart", action="store_false", dest="use_dart", help="Disable DART-specific tests.")
    parser.set_defaults(use_dart=True)
    args = parser.parse_args()

    dataset_cfg = load_dataset_cfg(args.dataset)
    mesh, _, mesh_rules = build_parallel(args.parallel)

    with jax.set_mesh(mesh):
        bundle = DynamicsCheckpointBundle.from_pretrained(args.ckpt, mesh_rules=mesh_rules, rngs=nnx.Rngs(0))
        latents, actions, step_idx, tau_idx = prepare_batch(bundle, dataset_cfg, args.seed)

        # Run tests
        if args.use_dart:
            test_dart_clean_independent_from_noisy(bundle, latents, actions, args.tol, args.seed + 1)
            test_dart_noisy_not_same_time_clean(bundle, latents, actions, args.tol, args.seed + 2)
            test_dart_noisy_not_other_noisy(bundle, latents, actions, args.tol, args.seed + 3)
        else:
            test_cached_vs_full_last(bundle, latents, actions, step_idx, tau_idx, args.tol)
            test_cached_vs_full_multistep(bundle, latents, actions, step_idx, tau_idx, args.tol)
            test_causal_truncation(bundle, latents, actions, step_idx, tau_idx, args.tol)

    print("All consistency tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
