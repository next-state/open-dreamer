"""
Test script for next_frame function in generation.py

This script tests the next_frame function by:
1. Loading the dynamics model and tokenizer from checkpoint
2. Initializing caches similar to reactor.py
3. Generating a sequence of frames with different actions
4. Saving output frames to verify visual quality
"""

import jax
import jax.numpy as jnp
import numpy as np
from dreamer.models import Tokenizer, Dynamics
from dreamer.generation import DenoiseSchedule, next_frame
import matplotlib.pyplot as plt
from pathlib import Path
import time

def test_next_frame_basic():
    """Test basic functionality of next_frame with single action."""
    print("=" * 80)
    print("TEST 1: Basic next_frame functionality")
    print("=" * 80)
    
    # Configuration (update this path to your checkpoint)
    DYNAMICS_CKPT = "logs/dynamics/checkpoints"
    
    # Load models similar to reactor.py
    print("\n1. Loading models from checkpoint...")
    start = time.time()
    dynamics, dynamics_vars, dynamics_cfg, tokenizer, tokenizer_vars, tokenizer_cfg = \
        Dynamics.from_pretrained(DYNAMICS_CKPT)
    print(f"   ✓ Models loaded in {time.time() - start:.2f}s")
    
    # Initialize denoising schedule
    print("\n2. Initializing denoising schedule...")
    num_steps = 4
    k_max = 256
    tau_ctx = 0.9
    schedule = DenoiseSchedule.init(num_steps=num_steps, k_max=k_max, tau_ctx=tau_ctx)
    print(f"   ✓ Schedule initialized: {num_steps} steps, k_max={k_max}, tau_ctx={tau_ctx}")
    
    # Compute latent shape (from reactor.py logic)
    print("\n3. Computing latent shape...")
    H, W = 64, 64  # CoinRun size
    patch_size = tokenizer_cfg.patch_size
    packing_factor = dynamics.config.packing_factor
    
    patches_h = H // patch_size
    patches_w = W // patch_size
    n_patches = patches_h * patches_w
    n_spatial = n_patches // packing_factor
    
    D_s = tokenizer_cfg.encoder.d_bottleneck
    latent_shape = (1, 1, n_spatial, D_s * packing_factor)
    print(f"   ✓ Latent shape: {latent_shape}")
    print(f"     - Image size: {H}x{W}")
    print(f"     - Patch size: {patch_size}")
    print(f"     - Packing factor: {packing_factor}")
    print(f"     - Spatial tokens: {n_spatial}")
    
    # Initialize caches
    print("\n4. Initializing KV caches...")
    window_size = tokenizer_cfg.dataset.T // packing_factor
    
    dynamics_cache = dynamics.create_static_caches(
        batch_size=1,
        n_spatial=n_spatial,
        window_size=window_size,
    )
    
    tokenizer_cache = tokenizer.create_static_caches(
        batch_size=1,
        window_size=window_size,
    )
    print(f"   ✓ Caches initialized (window_size={window_size})")
    
    # Test with a single action (no movement = 4 in CoinRun)
    print("\n5. Testing next_frame with action=4 (no movement)...")
    rng = jax.random.PRNGKey(42)
    action = jnp.array(4, dtype=jnp.int32)
    
    start = time.time()
    frame, h_last, dynamics_cache_updated, tokenizer_cache_updated, rng_updated = next_frame(
        tokenizer=tokenizer,
        tokenizer_vars=tokenizer_vars,
        dynamics=dynamics,
        dynamics_vars=dynamics_vars,
        schedule=schedule,
        action=action,
        latent_shape=latent_shape,
        dynamics_cache=dynamics_cache,
        tokenizer_cache=tokenizer_cache,
        rng=rng,
        task=None,
    )
    elapsed = time.time() - start
    
    # Verify outputs
    print(f"\n   ✓ Frame generated in {elapsed:.3f}s")
    print(f"     - Frame shape: {frame.shape} (expected: ({H}, {W}, 3))")
    print(f"     - Frame dtype: {frame.dtype} (expected: uint8)")
    print(f"     - Frame range: [{frame.min()}, {frame.max()}]")
    print(f"     - h_last is None: {h_last is None}")
    print(f"     - Caches updated: {dynamics_cache_updated is not None and tokenizer_cache_updated is not None}")
    
    assert frame.shape == (H, W, 3), f"Unexpected frame shape: {frame.shape}"
    assert frame.dtype == np.uint8, f"Unexpected frame dtype: {frame.dtype}"
    assert 0 <= frame.min() <= frame.max() <= 255, f"Frame values out of range"
    
    print("\n   ✅ TEST 1 PASSED")
    return frame, dynamics_cache_updated, tokenizer_cache_updated, rng_updated


def test_next_frame_sequence():
    """Test generating a sequence of frames with different actions."""
    print("\n" + "=" * 80)
    print("TEST 2: Sequential frame generation with different actions")
    print("=" * 80)
    
    # Configuration
    DYNAMICS_CKPT = "/home/ubuntu/dreamer4-jax-private/checkpoints/dynamics_latest"
    
    # Load models
    print("\n1. Loading models...")
    start = time.time()
    dynamics, dynamics_vars, dynamics_cfg, tokenizer, tokenizer_vars, tokenizer_cfg = \
        Dynamics.from_pretrained(DYNAMICS_CKPT)
    print(f"   ✓ Models loaded in {time.time() - start:.2f}s")
    
    # Initialize schedule
    schedule = DenoiseSchedule.init(num_steps=4, k_max=256, tau_ctx=0.9)
    
    # Compute latent shape
    H, W = 64, 64
    patch_size = tokenizer_cfg.patch_size
    packing_factor = dynamics.config.packing_factor
    n_patches = (H // patch_size) * (W // patch_size)
    n_spatial = n_patches // packing_factor
    D_s = tokenizer_cfg.encoder.d_bottleneck
    latent_shape = (1, 1, n_spatial, D_s * packing_factor)
    
    # Initialize caches
    window_size = tokenizer_cfg.dataset.T // packing_factor
    dynamics_cache = dynamics.create_static_caches(
        batch_size=1,
        n_spatial=n_spatial,
        window_size=window_size,
    )
    tokenizer_cache = tokenizer.create_static_caches(
        batch_size=1,
        window_size=window_size,
    )
    
    # Generate sequence of frames with different actions
    print("\n2. Generating sequence of 10 frames...")
    # CoinRun actions: 1=left, 7=right, 5=jump, 4=noop
    actions = [4, 4, 7, 7, 5, 7, 7, 1, 1, 4]  # Move right, jump, move left, stop
    
    frames = []
    rng = jax.random.PRNGKey(42)
    
    for i, action_idx in enumerate(actions):
        action = jnp.array(action_idx, dtype=jnp.int32)
        
        start = time.time()
        frame, h_last, dynamics_cache, tokenizer_cache, rng = next_frame(
            tokenizer=tokenizer,
            tokenizer_vars=tokenizer_vars,
            dynamics=dynamics,
            dynamics_vars=dynamics_vars,
            schedule=schedule,
            action=action,
            latent_shape=latent_shape,
            dynamics_cache=dynamics_cache,
            tokenizer_cache=tokenizer_cache,
            rng=rng,
            task=None,
        )
        elapsed = time.time() - start
        
        frames.append(frame)
        print(f"   Frame {i+1}/10: action={action_idx}, time={elapsed:.3f}s, "
              f"range=[{frame.min()}, {frame.max()}]")
    
    # Save frames as a grid
    print("\n3. Saving frames to output...")
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, (frame, action_idx) in enumerate(zip(frames, actions)):
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {i+1}\nAction: {action_idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "next_frame_sequence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved frame sequence to: {output_path}")
    
    print("\n   ✅ TEST 2 PASSED")
    return frames


def test_next_frame_cache_persistence():
    """Test that caches properly accumulate context over multiple frames."""
    print("\n" + "=" * 80)
    print("TEST 3: Cache persistence and context accumulation")
    print("=" * 80)
    
    # Configuration
    DYNAMICS_CKPT = "/home/ubuntu/dreamer4-jax-private/checkpoints/dynamics_latest"
    
    # Load models
    print("\n1. Loading models...")
    dynamics, dynamics_vars, dynamics_cfg, tokenizer, tokenizer_vars, tokenizer_cfg = \
        Dynamics.from_pretrained(DYNAMICS_CKPT)
    
    # Initialize
    schedule = DenoiseSchedule.init(num_steps=4, k_max=256, tau_ctx=0.9)
    H, W = 64, 64
    patch_size = tokenizer_cfg.patch_size
    packing_factor = dynamics.config.packing_factor
    n_patches = (H // patch_size) * (W // patch_size)
    n_spatial = n_patches // packing_factor
    D_s = tokenizer_cfg.encoder.d_bottleneck
    latent_shape = (1, 1, n_spatial, D_s * packing_factor)
    window_size = tokenizer_cfg.dataset.T // packing_factor
    
    # Test cache indices
    print("\n2. Testing cache index updates...")
    dynamics_cache = dynamics.create_static_caches(
        batch_size=1,
        n_spatial=n_spatial,
        window_size=window_size,
    )
    tokenizer_cache = tokenizer.create_static_caches(
        batch_size=1,
        window_size=window_size,
    )
    
    rng = jax.random.PRNGKey(42)
    action = jnp.array(4, dtype=jnp.int32)
    
    # Track cache indices
    for i in range(5):
        # Get initial cache index (from first time layer)
        dyn_cache_idx_before = dynamics_cache[0].index if dynamics_cache else None
        tok_cache_idx_before = tokenizer_cache[0].index if tokenizer_cache else None
        
        frame, h_last, dynamics_cache, tokenizer_cache, rng = next_frame(
            tokenizer=tokenizer,
            tokenizer_vars=tokenizer_vars,
            dynamics=dynamics,
            dynamics_vars=dynamics_vars,
            schedule=schedule,
            action=action,
            latent_shape=latent_shape,
            dynamics_cache=dynamics_cache,
            tokenizer_cache=tokenizer_cache,
            rng=rng,
            task=None,
        )
        
        dyn_cache_idx_after = dynamics_cache[0].index if dynamics_cache else None
        tok_cache_idx_after = tokenizer_cache[0].index if tokenizer_cache else None
        
        print(f"   Step {i+1}: Dynamics cache index: {dyn_cache_idx_before} → {dyn_cache_idx_after}, "
              f"Tokenizer cache index: {tok_cache_idx_before} → {tok_cache_idx_after}")
        
        # Verify cache indices increment
        if dyn_cache_idx_before is not None:
            assert dyn_cache_idx_after > dyn_cache_idx_before, "Dynamics cache not updating!"
        if tok_cache_idx_before is not None:
            assert tok_cache_idx_after > tok_cache_idx_before, "Tokenizer cache not updating!"
    
    print("\n   ✅ TEST 3 PASSED - Caches accumulating properly")


def test_next_frame_different_rngs():
    """Test that different RNG keys produce different frames."""
    print("\n" + "=" * 80)
    print("TEST 4: RNG determinism and variation")
    print("=" * 80)
    
    # Configuration
    DYNAMICS_CKPT = "/home/ubuntu/dreamer4-jax-private/checkpoints/dynamics_latest"
    
    # Load models
    print("\n1. Loading models...")
    dynamics, dynamics_vars, dynamics_cfg, tokenizer, tokenizer_vars, tokenizer_cfg = \
        Dynamics.from_pretrained(DYNAMICS_CKPT)
    
    # Initialize
    schedule = DenoiseSchedule.init(num_steps=4, k_max=256, tau_ctx=0.9)
    H, W = 64, 64
    patch_size = tokenizer_cfg.patch_size
    packing_factor = dynamics.config.packing_factor
    n_patches = (H // patch_size) * (W // patch_size)
    n_spatial = n_patches // packing_factor
    D_s = tokenizer_cfg.encoder.d_bottleneck
    latent_shape = (1, 1, n_spatial, D_s * packing_factor)
    window_size = tokenizer_cfg.dataset.T // packing_factor
    
    print("\n2. Testing RNG determinism (same seed should give same frame)...")
    
    # Generate with same RNG twice
    action = jnp.array(4, dtype=jnp.int32)
    
    for seed in [42, 123]:
        dynamics_cache = dynamics.create_static_caches(1, n_spatial, window_size)
        tokenizer_cache = tokenizer.create_static_caches(1, window_size)
        rng = jax.random.PRNGKey(seed)
        
        frame1, _, _, _, _ = next_frame(
            tokenizer, tokenizer_vars, dynamics, dynamics_vars, schedule,
            action, latent_shape, dynamics_cache, tokenizer_cache, rng, None
        )
        
        # Reset and regenerate
        dynamics_cache = dynamics.create_static_caches(1, n_spatial, window_size)
        tokenizer_cache = tokenizer.create_static_caches(1, window_size)
        rng = jax.random.PRNGKey(seed)
        
        frame2, _, _, _, _ = next_frame(
            tokenizer, tokenizer_vars, dynamics, dynamics_vars, schedule,
            action, latent_shape, dynamics_cache, tokenizer_cache, rng, None
        )
        
        diff = np.abs(frame1.astype(float) - frame2.astype(float)).mean()
        print(f"   Seed {seed}: Mean pixel difference = {diff:.6f}")
        assert diff < 0.01, f"Frames should be identical for same seed! Diff: {diff}"
    
    print("\n3. Testing RNG variation (different seeds should give different frames)...")
    
    # Generate with different RNGs
    dynamics_cache1 = dynamics.create_static_caches(1, n_spatial, window_size)
    tokenizer_cache1 = tokenizer.create_static_caches(1, window_size)
    rng1 = jax.random.PRNGKey(42)
    
    dynamics_cache2 = dynamics.create_static_caches(1, n_spatial, window_size)
    tokenizer_cache2 = tokenizer.create_static_caches(1, window_size)
    rng2 = jax.random.PRNGKey(999)
    
    frame1, _, _, _, _ = next_frame(
        tokenizer, tokenizer_vars, dynamics, dynamics_vars, schedule,
        action, latent_shape, dynamics_cache1, tokenizer_cache1, rng1, None
    )
    
    frame2, _, _, _, _ = next_frame(
        tokenizer, tokenizer_vars, dynamics, dynamics_vars, schedule,
        action, latent_shape, dynamics_cache2, tokenizer_cache2, rng2, None
    )
    
    diff = np.abs(frame1.astype(float) - frame2.astype(float)).mean()
    print(f"   Different seeds: Mean pixel difference = {diff:.6f}")
    assert diff > 1.0, f"Frames should differ for different seeds! Diff: {diff}"
    
    print("\n   ✅ TEST 4 PASSED - RNG working correctly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("NEXT_FRAME FUNCTION TEST SUITE")
    print("=" * 80)
    
    try:
        # Run all tests
        test_next_frame_basic()
        test_next_frame_sequence()
        test_next_frame_cache_persistence()
        test_next_frame_different_rngs()
        
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80)
        print("\nThe next_frame function is working correctly!")
        print("Check the 'test_outputs' directory for generated frame sequences.")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
