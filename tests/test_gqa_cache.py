#%%

import os
import sys
# Add project root to path to allow importing dreamer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from dreamer.models import GroupedQueryAttention, KVCache

def test_gqa_cache_consistency():
    print("Testing GroupedQueryAttention Cache Consistency...")
    
    # Test Parameters
    B = 2
    T = 16
    dim = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = dim // num_heads
    
    # Ensure dimensions match logic
    assert dim % num_heads == 0
    assert num_heads % num_kv_heads == 0
    
    key = jax.random.PRNGKey(42)
    key_init, key_input = jax.random.split(key)
    
    # Initialize Model with causal masking enabled
    model = GroupedQueryAttention(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        is_causal=True,
        dropout_rate=0.0,
        deterministic=True
    )
    
    # 1. Generate Input Data
    x = jax.random.normal(key_input, (B, T, dim))
    
    # Initialize variables (using full sequence)
    variables = model.init(key_init, x, mask=None)
    
    # --- Case 1: Full Sequence Forward Pass (No Cache) ---
    print("\nRunning full sequence forward pass (no cache)...")
    out_full, _ = model.apply(variables, x, mask=None)
    
    # --- Case 2: Prefill + Decode (With Cache) ---
    print("Running prefill + decode pass (with cache)...")
    
    # A. Prefill: all frames except the last one
    x_prefill = x[:, :-1, :]
    
    # Initialize Cache
    # usage: window_size should cover the sequence length we want to retain
    initial_cache = KVCache.init(
        batch_size=B,
        window_size=T,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=jnp.float32
    )
    
    # Run prefill
    out_prefill, cache_after_prefill = model.apply(
        variables, 
        x_prefill, 
        mask=None, 
        cache=initial_cache
    )
    
    # B. Decode: the last frame
    x_last = x[:, -1:, :] # Shape (B, 1, dim)
    
    # Run decode using the cache from prefill
    out_decode, cache_final = model.apply(
        variables, 
        x_last, 
        mask=None, 
        cache=cache_after_prefill
    )
    
    # --- 3. Comparison ---
    print("\nComparing results...")
    
    # Compare the output of the prefill phase with the corresponding part of full output
    # out_full: (B, T, dim)
    # out_prefill: (B, T-1, dim)
    prefill_diff = jnp.max(jnp.abs(out_full[:, :-1, :] - out_prefill))
    print(f"Prefill difference (max abs): {prefill_diff:.2e}")
    
    # Compare the output of the decode phase (last frame)
    # out_full last frame: (B, 1, dim)
    last_frame_full = out_full[:, -1:, :]
    # out_decode: (B, 1, dim)
    last_frame_decoded = out_decode
    
    decode_diff = jnp.max(jnp.abs(last_frame_full - last_frame_decoded))
    print(f"Decode difference (max abs): {decode_diff:.2e}")
    
    # Check if cache index is correct
    print(f"Final cache index: {cache_final.index}")
    expected_index = T
    assert cache_final.index == expected_index, f"Cache index mismatch: expected {expected_index}, got {cache_final.index}"
    
    # Assertions
    tol = 1e-5
    if prefill_diff < tol and decode_diff < tol:
        print("\nSUCCESS: Outputs match within tolerance!")
    else:
        print("\nFAILURE: Outputs do not match!")
        assert prefill_diff < tol, f"Prefill mismatch: {prefill_diff}"
        assert decode_diff < tol, f"Decode mismatch: {decode_diff}"

if __name__ == "__main__":
    test_gqa_cache_consistency()

# %%
