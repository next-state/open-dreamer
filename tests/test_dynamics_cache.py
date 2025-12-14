
import os
import sys
# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
import numpy as np
from dreamer.models import Dynamics

def test_dynamics_cache_consistency():
    print("Testing Dynamics Cache Consistency...")
    
    # Test Config
    B = 2
    T = 8
    
    d_model = 32
    d_bottleneck = 16
    d_spatial = 32
    n_spatial = 4
    n_register = 2
    n_agent = 1
    n_heads = 4
    n_kv_heads = 2
    depth = 2
    k_max = 8
    
    key = jax.random.PRNGKey(42)
    
    # Initialize Dynamics
    model = Dynamics(
        d_model=d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=n_register,
        n_agent=n_agent,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        depth=depth,
        k_max=k_max,
        dropout_rate=0.0,
        time_every=2  # Make sure we have time attention layers
    )
    
    # Generate Inputs
    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    
    actions = jax.random.randint(k1, (B, T), 0, 5)
    step_idxs = jax.random.randint(k2, (B, T), 0, 3)
    signal_idxs = jax.random.randint(k3, (B, T), 0, k_max)
    packed_enc_tokens = jax.random.normal(k4, (B, T, n_spatial, d_spatial))
    agent_tokens = jax.random.normal(k5, (B, T, n_agent, d_model))
    
    # Init variables
    variables = model.init(
        key, 
        actions, 
        step_idxs, 
        signal_idxs, 
        packed_enc_tokens, 
        agent_tokens=agent_tokens
    )
    
    # 1. Full Run (No Cache)
    print("\nRunning full sequence forward pass (no cache)...")
    x1_hat_full, h_full, _ = model.apply(
        variables,
        actions,
        step_idxs,
        signal_idxs,
        packed_enc_tokens,
        agent_tokens=agent_tokens,
        deterministic=True
    )
    
    # 2. Prefill + Decode (With Cache)
    print("Running prefill + decode pass (with cache)...")
    
    # A. Prefill (T-1)
    actions_pre = actions[:, :-1]
    step_pre = step_idxs[:, :-1]
    sig_pre = signal_idxs[:, :-1]
    packed_pre = packed_enc_tokens[:, :-1]
    agent_pre = agent_tokens[:, :-1]
    
    # Create static caches
    # We set window_size to T to accommodate the full sequence
    caches_init = model.apply(
        variables, 
        B, T, 
        method=model.create_static_caches
    )
    
    x1_hat_pre, h_pre, caches_after_pre = model.apply(
        variables,
        actions_pre,
        step_pre,
        sig_pre,
        packed_pre,
        agent_tokens=agent_pre,
        deterministic=True,
        caches=caches_init
    )
    
    # B. Decode (Last step)
    actions_last = actions[:, -1:]
    step_last = step_idxs[:, -1:]
    sig_last = signal_idxs[:, -1:]
    packed_last = packed_enc_tokens[:, -1:]
    agent_last = agent_tokens[:, -1:]
    
    x1_hat_dec, h_dec, caches_final = model.apply(
        variables,
        actions_last,
        step_last,
        sig_last,
        packed_last,
        agent_tokens=agent_last,
        deterministic=True,
        caches=caches_after_pre
    )
    
    # 3. Compare outputs
    print("\nComparing results...")
    
    # full: x1_hat (B, T, n_spatial, d_model) -> wait, check return shape
    # returns x1_hat (B, T, n_spatial, d_spatial) (projected back)??
    # No, flow_x_head maps to d_spatial.
    
    # Check Prefill match
    diff_x1_pre = jnp.max(jnp.abs(x1_hat_full[:, :-1] - x1_hat_pre))
    if n_agent > 0:
        diff_h_pre = jnp.max(jnp.abs(h_full[:, :-1] - h_pre))
    else:
        diff_h_pre = 0.0

    print(f"Prefill x1_hat diff: {diff_x1_pre:.2e}")
    print(f"Prefill h_t diff:    {diff_h_pre:.2e}")
    
    # Check Decode match
    diff_x1_dec = jnp.max(jnp.abs(x1_hat_full[:, -1:] - x1_hat_dec))
    if n_agent > 0:
        diff_h_dec = jnp.max(jnp.abs(h_full[:, -1:] - h_dec))
    else:
        diff_h_dec = 0.0
        
    print(f"Decode x1_hat diff:  {diff_x1_dec:.2e}")
    print(f"Decode h_t diff:     {diff_h_dec:.2e}")
    
    tol = 1e-4
    if diff_x1_dec < tol and diff_h_dec < tol:
        print("\nSUCCESS: Dynamics cache outputs match!")
    else:
        print("\nFAILURE: Dynamics cache mismatch!")
        # Print cache info
        if caches_final:
            first_idx = list(caches_final.keys())[0]
            print(f"Final cache index (layer {first_idx}): {caches_final[first_idx].index}")
            print(f"Expected index: {T}")

if __name__ == "__main__":
    test_dynamics_cache_consistency()
