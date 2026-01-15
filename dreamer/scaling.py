"""Scaling law utilities for compute-optimal training experiments.

This module provides training step computation following the methodology
from nanochat/Chinchilla scaling laws. FLOP estimation has been moved to
model classes (Tokenizer.estimate_flops(), Dynamics.estimate_flops()).

References:
- https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
- https://github.com/karpathy/nanochat
"""


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
) -> int:
    """Compute training steps to achieve a fixed FLOPs budget.

    For iso-FLOPs experiments: train different model sizes for different
    durations such that total compute is constant.

    Args:
        total_flops: Target total FLOPs budget
        flops_per_step: FLOPs per training step for this model

    Returns:
        Number of training steps
    """
    return int(total_flops / flops_per_step)
