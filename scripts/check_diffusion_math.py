#!/usr/bin/env python3
"""
Quick sanity check for the tau-ladder update math.

Compares:
  1) The in-code update used in dreamer/generation.py
  2) The closed-form update that reconstructs z0 and re-mixes

If these match (they should), the per-step math is consistent.
"""
import argparse

import jax
import jax.numpy as jnp

from dreamer.generation import DenoiseSchedule


def update_alpha(z_t, x_pred, tau_prev, tau_curr):
    alpha = (tau_curr - tau_prev) / jnp.maximum(1.0 - tau_prev, 1e-8)
    return (1.0 - alpha) * z_t + alpha * x_pred


def update_closed_form(z_t, x_pred, tau_prev, tau_curr):
    # Reconstruct the original noise z0, then re-mix at tau_curr.
    z0 = (z_t - tau_prev * x_pred) / jnp.maximum(1.0 - tau_prev, 1e-8)
    return (1.0 - tau_curr) * z0 + tau_curr * x_pred


def parse_shape(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("shape must be a comma-separated list of ints, e.g. 2,1,8,64")
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description="Check tau-ladder update math.")
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--k-max", type=int, default=256)
    parser.add_argument("--shape", type=str, default="2,1,8,64")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    schedule = DenoiseSchedule.init(num_steps=args.num_steps, k_max=args.k_max)
    shape = parse_shape(args.shape)

    key = jax.random.PRNGKey(args.seed)
    key, k1, k2 = jax.random.split(key, 3)
    z_t = jax.random.normal(k1, shape, dtype=jnp.float32)
    x_pred = jax.random.normal(k2, shape, dtype=jnp.float32)

    max_diff = 0.0
    mean_diff = 0.0
    for s in range(schedule.num_steps):
        tau_prev = schedule.tau_values[s]
        tau_curr = schedule.tau_values[s + 1]
        out_a = update_alpha(z_t, x_pred, tau_prev, tau_curr)
        out_b = update_closed_form(z_t, x_pred, tau_prev, tau_curr)
        diff = jnp.abs(out_a - out_b)
        max_diff = float(jnp.maximum(max_diff, jnp.max(diff)))
        mean_diff = float(jnp.maximum(mean_diff, jnp.mean(diff)))

    print(f"num_steps={args.num_steps} k_max={args.k_max} shape={shape}")
    print(f"max_abs_diff={max_diff:.3e} mean_abs_diff={mean_diff:.3e}")


if __name__ == "__main__":
    main()
