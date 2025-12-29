# export XLA_PYTHON_CLIENT_PREALLOCATE=false && uv run scripts/benchmark_tokenizer.py ++dataset.B=8 ++dataset.T=4 ++max_steps=5

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict
from typing import Dict, Any

from dreamer.configs import TokenizerConfig
from dreamer.models import Tokenizer
from dreamer.utils import init_tokenizer, to_jnp_dtype
from scripts.train_tokenizer import train_step, forward_apply

# Suppress absl info logs
import logging
logging.getLogger('absl').setLevel(logging.WARNING)

def measure_memory():
    """Get current memory usage in GB."""
    # This works for GPU/TPU
    try:
        stats = jax.devices()[0].memory_stats()
        return stats['bytes_in_use'] / (1024**3)
    except:
        return 0.0

def benchmark_precision(cfg: TokenizerConfig, dtype_str: str, num_steps: int = 20):
    print(f"\n>>> Benchmarking {dtype_str}...")
    
    # Update config for the specific precision
    cfg.encoder.dtype = dtype_str
    cfg.encoder.param_dtype = "float32" # Keep params in fp32 for master weights if desired, or match
    cfg.decoder.dtype = dtype_str
    cfg.decoder.param_dtype = "float32"

    rng = jax.random.PRNGKey(0)
    
    # Data init (dummy)
    B, T, H, W, C = cfg.dataset.B, cfg.dataset.T, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C
    dummy_videos = jax.random.uniform(rng, (B, T, H, W, C), dtype=jnp.float32)
    
    # Model init
    tokenizer = Tokenizer(cfg)
    rng, variables = init_tokenizer(rng, tokenizer, cfg)
    params = variables["params"]
    
    tx = optax.adamw(cfg.lr)
    opt_state = tx.init(params)
    
    apply_fn = tokenizer.apply
    
    # Pre-warmup / Compile
    print("Compiling...")
    start_compile = time.time()
    params, opt_state, aux = train_step(
        apply_fn, tx, variables, params, opt_state, dummy_videos,
        master_key=rng, step=0, 
        lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac,
        dataset_mean=tuple(cfg.dataset.dataset_mean),
        dataset_std=tuple(cfg.dataset.dataset_std),
        log_gradients=False,
        tokenizer_loss_type=cfg.tokenizer_loss_type
    )
    # block_until_ready is important for timing
    jax.block_until_ready(params)
    compile_time = time.time() - start_compile
    print(f"Compile time: {compile_time:.2f}s")
    
    # Measure memory after compilation and first step
    mem_used = measure_memory()
    print(f"Memory in use: {mem_used:.2f} GB")
    
    # Actual benchmark loop
    print(f"Running {num_steps} steps...")
    start_loop = time.time()
    for i in range(1, num_steps + 1):
        params, opt_state, aux = train_step(
            apply_fn, tx, variables, params, opt_state, dummy_videos,
            master_key=rng, step=i, 
            lpips_weight=cfg.lpips_weight, lpips_frac=cfg.lpips_frac,
            dataset_mean=tuple(cfg.dataset.dataset_mean),
            dataset_std=tuple(cfg.dataset.dataset_std),
            log_gradients=False,
            tokenizer_loss_type=cfg.tokenizer_loss_type
        )
    jax.block_until_ready(params)
    total_loop_time = time.time() - start_loop
    avg_step_time = total_loop_time / num_steps
    print(f"Average step time: {avg_step_time*1000:.2f} ms")
    
    return {
        "precision": dtype_str,
        "compile_time": compile_time,
        "avg_step_time_ms": avg_step_time * 1000,
        "memory_gb": mem_used
    }

@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: DictConfig):
    # Register resolver if not already done (it might be done in train_tokenizer import but just in case)
    try:
        OmegaConf.register_new_resolver("mul", lambda *args: __import__('functools').reduce(__import__('operator').mul, args))
    except:
        pass
        
    schema = OmegaConf.structured(TokenizerConfig)
    cfg = OmegaConf.merge(schema, cfg)
    base_cfg = OmegaConf.to_object(cfg)
    
    results = []
    
    # Run Float32
    fp32_res = benchmark_precision(base_cfg, "float32")
    results.append(fp32_res)
    
    # Run BFloat16
    # We clear caches and try to free memory between runs
    # Clearing caches is hard in JIT without re-running script, 
    # but we'll try to at least clear the variables.
    
    bf16_res = benchmark_precision(base_cfg, "bfloat16")
    results.append(bf16_res)
    
    # Summary Table
    print("\n" + "="*50)
    print(f"{'Precision':<15} | {'Step Time (ms)':<15} | {'Memory (GB)':<12}")
    print("-" * 50)
    for res in results:
        print(f"{res['precision']:<15} | {res['avg_step_time_ms']:<15.2f} | {res['memory_gb']:<12.2f}")
    print("="*50)

    # Calculate gains
    t32 = fp32_res['avg_step_time_ms']
    t16 = bf16_res['avg_step_time_ms']
    m32 = fp32_res['memory_gb']
    m16 = bf16_res['memory_gb']
    
    speedup = (t32 / t16 - 1) * 100
    mem_saving = (1 - m16 / m32) * 100 if m32 > 0 else 0
    
    print(f"Speedup: {speedup:.1f}%")
    print(f"Memory Saving: {mem_saving:.1f}%")

if __name__ == "__main__":
    main()
