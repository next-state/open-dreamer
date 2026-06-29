"""Compare observed-frame cache prefill batched vs frame-by-frame.

This is a CPU-friendly diagnostic for reactor_app.pipeline._observe_frame.
It extracts an MP4 payload from an ArrayRecord shard, decodes a short frame
sequence, then compares:

1. one batched tokenizer/dynamics/decoder cache update over T frames
2. T calls to the same single-frame observed-frame path used by Reactor

The dynamics observation noise is precomputed from the same per-frame split
sequence used by _observe_frame so the two paths receive identical tensors.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import decord
import grain
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from dreamer.actions import NUM_BINARY_ACTIONS, NUM_CAMERA_CLASSES, Actions, parse_action_dicts
from dreamer.configs import DecoderModelConfig, DynamicsModelConfig, EncoderModelConfig, TokenizerModelConfig
from dreamer.generation import DenoiseSchedule
from dreamer.models import Dynamics, Tokenizer
from dreamer.parallel import MeshRules
from dreamer.utils import normalize_latents
from reactor_app.pipeline import _observe_frame


def _extract_mp4(array_record: Path, record_index: int, out_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    source = grain.sources.ArrayRecordDataSource([str(array_record)])
    data = pickle.loads(source[record_index])
    video_bytes = data["video"]
    if not video_bytes.startswith(b"\x00\x00\x00") or b"ftyp" not in video_bytes[:32]:
        raise ValueError(f"Record {record_index} in {array_record} does not look like MP4 bytes")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(video_bytes)
    return out_path, data.get("actions", [])


def _load_frames(mp4_path: Path, num_frames: int, image_size: int) -> np.ndarray:
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(mp4_path), ctx=decord.cpu(0), num_threads=1)
    if len(vr) < num_frames:
        raise ValueError(f"{mp4_path} only has {len(vr)} frames, need {num_frames}")
    frames = vr.get_batch(list(range(num_frames))).asnumpy()
    resized = []
    resample = getattr(Image, "Resampling", Image).BILINEAR
    for frame in frames:
        resized.append(np.asarray(Image.fromarray(frame).resize((image_size, image_size), resample)))
    return np.ascontiguousarray(np.stack(resized, axis=0).astype(np.uint8, copy=False))


def _mesh_context(mesh: Any):
    if hasattr(jax, "set_mesh"):
        return jax.set_mesh(mesh)
    return mesh


def _make_models(image_size: int, context_length: int) -> tuple[Tokenizer, Dynamics, int, int, Any]:
    n_latents = 4
    d_bottleneck = 8
    patch_size = 8
    d_model = 32
    n_heads = 4

    tokenizer_cfg = TokenizerModelConfig(
        encoder=EncoderModelConfig(
            n_latents=n_latents,
            d_bottleneck=d_bottleneck,
            depth=2,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,
            patch_size=patch_size,
            dropout_rate=0.0,
            time_every=1,
            time_layer_offset=0,
            context_length=context_length,
            dtype="float32",
            param_dtype="float32",
        ),
        decoder=DecoderModelConfig(
            n_latents=n_latents,
            d_bottleneck=d_bottleneck,
            depth=2,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=1,
            patch_size=patch_size,
            d_patch=patch_size * patch_size * 3,
            dropout_rate=0.0,
            time_every=1,
            time_layer_offset=0,
            context_length=context_length,
            H=image_size,
            W=image_size,
            dtype="float32",
            param_dtype="float32",
        ),
    )
    dynamics_cfg = DynamicsModelConfig(
        d_bottleneck=d_bottleneck,
        depth=2,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=1,
        packing_factor=2,
        n_register=2,
        qk_norm_type=None,
        time_every=1,
        time_layer_offset=0,
        dropout_rate=0.0,
        k_max=8,
        context_length=context_length,
        num_binary_actions=NUM_BINARY_ACTIONS,
        categorical_action_dim=NUM_CAMERA_CLASSES,
        continuous_action_dim=0,
        dtype="float32",
        param_dtype="float32",
    )
    mesh = jax.make_mesh((1, 1), ("data", "model"))
    mesh_rules = MeshRules(embed=None, mlp="model", attn="model", data="data")
    rngs = nnx.Rngs(0)
    with _mesh_context(mesh):
        tokenizer = Tokenizer(tokenizer_cfg, mesh_rules=mesh_rules, rngs=rngs)
        dynamics = Dynamics(dynamics_cfg, mesh_rules=mesh_rules, rngs=rngs)
    return tokenizer, dynamics, n_latents, d_bottleneck, mesh


def _slice_actions(raw_actions: list[dict[str, Any]], num_frames: int) -> Actions:
    parsed = parse_action_dicts(raw_actions[:num_frames])
    return Actions(
        binary=jnp.asarray(parsed.binary[None, :num_frames], dtype=jnp.int32),
        categorical=jnp.asarray(parsed.categorical[None, :num_frames], dtype=jnp.int32),
        continuous=None,
    )


def _action_at(actions: Actions, index: int) -> Actions:
    return Actions(
        binary=actions.binary[:, index],
        categorical=actions.categorical[:, index],
        continuous=None,
    )


def _per_frame_noise(seed: int, num_frames: int, latent_shape: tuple[int, ...], dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    rng = jax.random.PRNGKey(seed)
    noises = []
    for _ in range(num_frames):
        rng, rng_ctx = jax.random.split(rng)
        noises.append(jax.random.normal(rng_ctx, shape=latent_shape, dtype=dtype))
    return jnp.concatenate(noises, axis=1), rng


def _observe_sequence_batched(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    frames: jax.Array,
    actions: Actions,
    dynamics_cache: Any,
    tokenizer_cache: Any,
    noise: jax.Array,
) -> tuple[Any, Any]:
    frames = jnp.asarray(frames, dtype=jnp.float32)[None, ...]
    latent, _aux, encoder_cache = tokenizer.encode(frames, deterministic=True, caches=tokenizer_cache["encoder"])
    latent_norm = normalize_latents(latent, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
    batch_size, num_frames = latent_norm.shape[:2]
    step_indices = jnp.full((batch_size, num_frames), schedule.step_idx_ctx, dtype=jnp.int32)
    tau_indices = jnp.full((batch_size, num_frames), schedule.tau_idx_ctx, dtype=jnp.int32)
    latent_noised = latent_norm * schedule.tau_ctx + (1.0 - schedule.tau_ctx) * noise

    _pred, (_h, dynamics_cache) = dynamics(
        actions,
        step_indices,
        tau_indices,
        latent_noised,
        deterministic=True,
        caches=dynamics_cache,
    )
    _decoded, decoder_cache = tokenizer.decode(latent, caches=tokenizer_cache["decoder"], deterministic=True)
    return dynamics_cache, {"encoder": encoder_cache, "decoder": decoder_cache}


def _tree_max_abs_diff(a: Any, b: Any) -> float:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    if len(leaves_a) != len(leaves_b):
        raise ValueError(f"Tree leaf counts differ: {len(leaves_a)} vs {len(leaves_b)}")
    max_diff = 0.0
    for left, right in zip(leaves_a, leaves_b, strict=True):
        left_arr = jnp.asarray(left)
        right_arr = jnp.asarray(right)
        if left_arr.shape != right_arr.shape:
            raise ValueError(f"Leaf shapes differ: {left_arr.shape} vs {right_arr.shape}")
        if left_arr.size:
            diff = jnp.max(jnp.abs(left_arr.astype(jnp.float32) - right_arr.astype(jnp.float32)))
            max_diff = max(max_diff, float(diff))
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-record", type=Path, default=Path("data/shard-00000.array_record"))
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--mp4-out", type=Path, default=Path("/tmp/dreamer_observe_cache_fixture.mp4"))
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.num_frames < 1:
        raise ValueError("--num-frames must be positive")
    context_length = args.context_length if args.context_length > 0 else max(8, args.num_frames)
    if context_length < 1:
        raise ValueError("--context-length must be positive when provided")
    if args.num_frames > context_length:
        raise ValueError(
            "The cached all-at-once comparison requires --num-frames <= --context-length. "
            "The Reactor observed-frame path streams T=1 updates and can run longer than "
            "the cache window; a single cached batched call cannot represent that case."
        )

    mp4_path, raw_actions = _extract_mp4(args.array_record, args.record_index, args.mp4_out)
    frames_np = _load_frames(mp4_path, args.num_frames, args.image_size)
    actions = _slice_actions(raw_actions, args.num_frames)
    tokenizer, dynamics, n_latents, d_bottleneck, mesh = _make_models(args.image_size, context_length)
    with _mesh_context(mesh):
        schedule = DenoiseSchedule.init(num_steps=2, k_max=dynamics.cfg.k_max, tau_ctx_target=0.75)

        dynamics_cache_batched = dynamics.create_static_caches(
            batch_size=1,
            n_latents=n_latents,
            window_size=context_length,
            n_agent=0,
            dtype=jnp.float32,
        )
        tokenizer_cache_batched = tokenizer.create_tokenizer_static_caches(
            batch_size=1,
            H=args.image_size,
            W=args.image_size,
            window_size=context_length,
            dtype=jnp.float32,
        )
        dynamics_cache_loop = dynamics_cache_batched
        tokenizer_cache_loop = tokenizer_cache_batched

        noise, expected_rng = _per_frame_noise(
            args.seed,
            args.num_frames,
            (1, 1, n_latents, d_bottleneck),
            jnp.float32,
        )
        dynamics_cache_batched, tokenizer_cache_batched = _observe_sequence_batched(
            tokenizer,
            dynamics,
            schedule,
            jnp.asarray(frames_np),
            actions,
            dynamics_cache_batched,
            tokenizer_cache_batched,
            noise,
        )

        rng = jax.random.PRNGKey(args.seed)
        for index in range(args.num_frames):
            dynamics_cache_loop, tokenizer_cache_loop, rng = _observe_frame(
                tokenizer,
                dynamics,
                schedule,
                jnp.asarray(frames_np[index]),
                _action_at(actions, index),
                dynamics_cache_loop,
                tokenizer_cache_loop,
                rng,
            )

        jax.block_until_ready((dynamics_cache_batched, tokenizer_cache_batched, dynamics_cache_loop, tokenizer_cache_loop))

        dynamics_diff = _tree_max_abs_diff(dynamics_cache_batched, dynamics_cache_loop)
        tokenizer_diff = _tree_max_abs_diff(tokenizer_cache_batched, tokenizer_cache_loop)
        rng_diff = _tree_max_abs_diff(expected_rng, rng)

    print(f"mp4_fixture={mp4_path}")
    print(f"frames_shape={frames_np.shape}")
    print(f"context_length={context_length}")
    print(f"dynamics_cache_max_abs_diff={dynamics_diff:.9g}")
    print(f"tokenizer_cache_max_abs_diff={tokenizer_diff:.9g}")
    print(f"rng_max_abs_diff={rng_diff:.9g}")
    print(f"allclose_1e-5={dynamics_diff <= 1e-5 and tokenizer_diff <= 1e-5 and rng_diff == 0.0}")


if __name__ == "__main__":
    main()
