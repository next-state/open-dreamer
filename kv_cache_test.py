"""
Throwaway test to verify that Tokenizer decoding with and without KV caching produces identical outputs.
"""

import jax
import jax.numpy as jnp
from dreamer.models import Tokenizer
from dreamer.configs import TokenizerConfig

def test_tokenizer_cache_consistency():
    """Test that decoding with and without cache produces the same output."""
    
    # Create a simple tokenizer config
    config = TokenizerConfig('asd')
    
    # Initialize tokenizer
    tokenizer = Tokenizer(config)
    
    # Create dummy input
    batch_size = 2
    T = 4  # temporal dimension
    rng = jax.random.PRNGKey(42)
    rng_params, rng_input = jax.random.split(rng)
    
    # Create dummy video input for initialization
    dummy_video = jax.random.uniform(
        rng_input, 
        (batch_size, T, config.dataset.H, config.dataset.W, config.dataset.C)
    )
    
    # Initialize model
    variables = tokenizer.init(
        {"params": rng_params, "mae": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        dummy_video,
        deterministic=True
    )
    
    # Encode video to get latents
    rng_mae, rng_dropout = jax.random.split(jax.random.PRNGKey(3))
    z, aux = tokenizer.apply(
        variables,
        dummy_video,
        deterministic=True,
        rngs={"mae": rng_mae, "dropout": rng_dropout},
        method=tokenizer.encode
    )
    
    print(f"Latent shape: {z.shape}")
    
    # Test 1: Decode without cache (full sequence at once)
    print("\n=== Test 1: Decoding without cache ===")
    rng_dec1, rng_dec2 = jax.random.split(jax.random.PRNGKey(4))
    output_no_cache, _ = tokenizer.apply(
        variables,
        z,
        deterministic=True,
        rngs={"dropout": rng_dec1},
        method=tokenizer.decode
    )
    print(f"Output shape (no cache): {output_no_cache.shape}")
    
    # Test 2: Decode with cache (simulating autoregressive decoding)
    print("\n=== Test 2: Decoding with cache (autoregressive) ===")
    
    # Create static caches
    n_patches = (config.decoder.H // config.decoder.patch_size) * \
                (config.decoder.W // config.decoder.patch_size)
    S_total = config.decoder.n_latents + n_patches
    
    caches = tokenizer.apply(
        variables,
        batch_size=batch_size,
        window_size=T,
        method=tokenizer.create_static_caches
    )
    
    print(f"Created {len(caches)} cache entries")
    
    # Decode with cache (full sequence, using cache for efficiency)
    # The cache should produce the same result as non-cached when given the full sequence
    rng_cache = jax.random.PRNGKey(4)  # Same RNG as no-cache version
    output_with_cache, final_caches = tokenizer.apply(
        variables,
        z,
        deterministic=True,
        caches=caches,
        rngs={"dropout": rng_cache},
        method=tokenizer.decode
    )
    print(f"Final output shape (with cache): {output_with_cache.shape}")
    
    # Test 3: Compare outputs
    print("\n=== Test 3: Comparing outputs ===")
    max_diff = jnp.max(jnp.abs(output_no_cache - output_with_cache))
    mean_diff = jnp.mean(jnp.abs(output_no_cache - output_with_cache))
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if they're close enough (accounting for numerical precision)
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"✓ PASS: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"✗ FAIL: Outputs differ by more than tolerance ({tolerance})")
        print(f"\nSample differences at first location:")
        print(f"No cache: {output_no_cache[0, 0, 0, 0, :]}")
        print(f"With cache: {output_with_cache[0, 0, 0, 0, :]}")
        return False

if __name__ == "__main__":
    print("Testing Tokenizer KV Cache Consistency\n")
    success = test_tokenizer_cache_consistency()
    print(f"\n{'='*50}")
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print(f"{'='*50}")
