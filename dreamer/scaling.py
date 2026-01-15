"""Scaling law utilities for compute-optimal training experiments.

This module provides FLOPs estimation and training step computation
following the methodology from nanochat/Chinchilla scaling laws.

Uses the Karpathy/Bahdanau formula: C = 6ND + attention_compute
- 6 FLOPs per parameter per token (covers all weight matrices)
- Plus attention computation FLOPs (Q@K^T and attn@V)

References:
- https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
- https://github.com/karpathy/nanochat
"""

# FIXME: use https://docs.jax.dev/en/latest/aot.html

def estimate_axial_transformer_flops(
    nparams: int,
    depth: int,
    d_model: int,
    batch_size: int,
    seq_length: int,
    n_spatial: int,
    time_every: int = 4,
) -> int:
    """Estimate training FLOPs for axial transformer.

    Uses Karpathy/Bahdanau methodology:
    - 6 FLOPs per parameter per token (covers all weight matrices)
    - Plus attention computation FLOPs (Q@K^T and attn@V)

    This codebase uses axial attention (alternating space/time layers):
    - Space layers: attention over S tokens, (B×T) parallel ops
    - Time layers: attention over T tokens, (B×S) parallel ops
    - Layer i is a time layer if (i+1) % time_every == 0

    Args:
        nparams: Number of model parameters
        depth: Number of transformer layers
        d_model: Model hidden dimension
        batch_size: Batch size (B)
        seq_length: Sequence length in time (T)
        n_spatial: Number of spatial tokens per frame (S)
        time_every: Time layer frequency (default 4)

    Returns:
        Estimated FLOPs per training step
    """
    B, T, S = batch_size, seq_length, n_spatial
    total_tokens = B * T * S

    # Weight FLOPs: all linear layers (Q,K,V,O projections + FFN)
    # 6 = 2 (forward) + 4 (backward) FLOPs per weight per token
    weight_flops = 6 * nparams * total_tokens

    # Attention computation FLOPs (Q@K^T and attn@V - not weight matrices)
    # 12 = 2 matmuls × 2 FLOPs × 3 (forward + backward)
    n_time_layers = depth // time_every
    n_space_layers = depth - n_time_layers
    space_attn = 12 * n_space_layers * d_model * (S ** 2) * B * T
    time_attn = 12 * n_time_layers * d_model * (T ** 2) * B * S

    return int(weight_flops + space_attn + time_attn)


def estimate_tokenizer_flops(
    nparams: int,
    encoder_depth: int,
    decoder_depth: int,
    d_model: int,
    batch_size: int,
    seq_length: int,
    n_patches: int,
    n_latents: int,
    time_every: int = 4,
) -> int:
    """Estimate FLOPs for tokenizer (encoder + decoder) with axial attention.

    Args:
        nparams: Total number of model parameters
        encoder_depth: Number of encoder transformer layers
        decoder_depth: Number of decoder transformer layers
        d_model: Model hidden dimension
        batch_size: Batch size (B)
        seq_length: Sequence length (T)
        n_patches: Number of patches per frame
        n_latents: Number of latent tokens
        time_every: Time layer frequency (default 4)

    Returns:
        Estimated FLOPs per training step
    """
    S = n_patches + n_latents
    total_depth = encoder_depth + decoder_depth

    return estimate_axial_transformer_flops(
        nparams, total_depth, d_model, batch_size, seq_length, S, time_every
    )


def estimate_dynamics_flops(
    nparams: int,
    depth: int,
    d_model: int,
    batch_size: int,
    seq_length: int,
    n_spatial: int,
    n_register: int,
    time_every: int = 4,
) -> int:
    """Estimate FLOPs for dynamics model with axial attention.

    Args:
        nparams: Total number of model parameters
        depth: Number of transformer layers
        d_model: Model hidden dimension
        batch_size: Batch size (B)
        seq_length: Sequence length (T)
        n_spatial: Number of spatial tokens per frame
        n_register: Number of register tokens
        time_every: Time layer frequency (default 4)

    Returns:
        Estimated FLOPs per training step
    """
    # S = action(1) + signal(1) + step(1) + spatial + registers
    S = 3 + n_spatial + n_register

    return estimate_axial_transformer_flops(
        nparams, depth, d_model, batch_size, seq_length, S, time_every
    )


def compute_max_steps(
    param_count: int,
    tokens_per_param: float,
    tokens_per_step: int,
) -> int:
    """Compute training steps from parameter count using fixed ratio.

    Following nanochat methodology: total_tokens = param_count × tokens_per_param

    Args:
        param_count: Total number of model parameters
        tokens_per_param: Target tokens per parameter ratio (e.g., 8.0)
        tokens_per_step: Tokens processed per training step (B × T)

    Returns:
        Number of training steps
    """
    total_tokens = param_count * tokens_per_param
    return int(total_tokens / tokens_per_step)


def compute_steps_for_flops_budget(
    total_flops: float,
    flops_per_step: int,
    min_steps: int = 10,
) -> int:
    """Compute training steps to achieve a fixed FLOPs budget.

    For iso-FLOPs experiments: train different model sizes for different
    durations such that total compute is constant.

    Args:
        total_flops: Target total FLOPs budget
        flops_per_step: FLOPs per training step for this model
        min_steps: Minimum steps to run (default 100)

    Returns:
        Number of training steps
    """
    steps = int(total_flops / flops_per_step)
    return max(steps, min_steps)
