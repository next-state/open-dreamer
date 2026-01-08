"""
Mu-Parameterization (μP) utilities for Dreamer 4.

μP ensures optimal hyperparameters transfer across model widths by controlling
the magnitude of activations, gradients, and weight updates so they don't scale
with model width.

Reference: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
Hyperparameter Transfer" (Yang et al., 2022)

Key principles:
- Hidden layer initialization: σ = 1/√fan_in (lecun_normal)
- Output layer initialization: σ = 1/width (smaller than hidden)
- Attention scaling: 1/d_head (not 1/√d_head)
- Learning rates: hidden layers get lr/m_d, embeddings get constant lr
  where m_d = d_model / base_width
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import jax
import jax.numpy as jnp
from flax import nnx
import optax


@dataclass(frozen=False)
class MuPConfig:
    """Configuration for mu-Parameterization.

    Attributes:
        enabled: Whether to use μP scaling. If False, standard parameterization is used.
        base_width: Reference width for computing the width multiplier m_d = d_model / base_width.
                   Set this to the width at which you tune hyperparameters.
        attn_scale_type: "mup" for 1/d_head scaling, "standard" for 1/√d_head.
    """
    enabled: bool = False
    base_width: int = 256
    attn_scale_type: str = "mup"  # "mup" or "standard"


def compute_width_multiplier(d_model: int, base_width: int) -> float:
    """Compute the width multiplier m_d = d_model / base_width."""
    return d_model / base_width


def mup_attention_scale(d_head: int, mup_enabled: bool = True) -> float:
    """
    Compute attention scaling factor.

    For μP: scale = 1/d_head (not 1/√d_head as in standard parameterization).
    This ensures dot products don't grow with width.

    Args:
        d_head: Dimension per attention head
        mup_enabled: If True, use 1/d_head; if False, use 1/√d_head

    Returns:
        Scaling factor for attention logits
    """
    if mup_enabled:
        return 1.0 / d_head
    else:
        return d_head ** -0.5


def mup_output_init(d_model: int, base_width: int) -> nnx.Initializer:
    """
    Create initializer for output/readout layers with μP scaling.

    Output layers should be initialized smaller than hidden layers:
    σ = 1 / m_d = base_width / d_model

    This ensures output activations don't scale with width.

    Args:
        d_model: Model hidden dimension
        base_width: Reference width for scaling

    Returns:
        nnx.Initializer with appropriate std
    """
    m_d = compute_width_multiplier(d_model, base_width)
    stddev = 1.0 / m_d
    return nnx.initializers.normal(stddev=stddev)


def mup_embed_init(stddev: float = 0.02) -> nnx.Initializer:
    """
    Create initializer for embeddings with constant std (not scaled by width).

    Embeddings have a single infinite dimension (vocab × d_model), so they
    should use constant initialization std regardless of width.
    """
    return nnx.initializers.normal(stddev=stddev)


def classify_param_for_mup(path: tuple, leaf: Any) -> str:
    """
    Classify a parameter path into μP categories for learning rate scaling.

    Categories:
        - "embedding": Embeddings and learned tokens (constant LR)
        - "hidden": Hidden layer weights (LR / m_d)
        - "output": Output projection weights (LR / m_d)
        - "other": Norms, biases, etc. (constant LR, no weight decay)

    Args:
        path: Tuple path to the parameter in the pytree
        leaf: The parameter array (unused, but required by tree_map_with_path)

    Returns:
        Category string for this parameter
    """
    path_str = '.'.join(str(p) for p in path).lower()

    # Output layers - need LR / m_d and smaller init
    output_patterns = [
        'to_out',           # Attention output projection
        'fc_out',           # MLP output projection
        'patch_head',       # Decoder patch head
        'bottleneck_proj',  # Encoder bottleneck projection
        # Note: flow_x_head uses zero init, handled separately
    ]
    for pattern in output_patterns:
        if pattern in path_str:
            return "output"

    # Embeddings - constant LR
    embed_patterns = [
        'emb',              # General embeddings
        'embed',            # Embedding layers
        'register_tokens',  # Register tokens
        'latents_enc',      # Encoder latents
        'patch_queries',    # Decoder patch queries
        'mask_token',       # MAE mask token
        'base_action_emb',  # Action encoder base
        'agent_base',       # Agent base embedding
    ]
    for pattern in embed_patterns:
        if pattern in path_str:
            return "embedding"

    # Normalization layers - constant LR, no weight decay
    if 'norm' in path_str or 'scale' in path_str:
        return "other"

    # Default: hidden layers (kernels)
    if 'kernel' in path_str:
        return "hidden"

    # Everything else (biases, etc.)
    return "other"


def create_mup_label_fn(mup_config: MuPConfig):
    """
    Create a function that labels parameters for μP-aware optimization.

    Returns a function compatible with optax.multi_transform.
    """
    def label_fn(params):
        return jax.tree_util.tree_map_with_path(classify_param_for_mup, params)
    return label_fn


def create_mup_optimizer(
    base_lr: float,
    d_model: int,
    mup_config: MuPConfig,
    weight_decay: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.9,
    schedule_fn: Optional[Callable[[int], float]] = None,
) -> optax.GradientTransformation:
    """
    Create an optimizer with μP-aware learning rate scaling.

    When μP is enabled, different parameter categories get different learning rates:
    - Embeddings: base_lr (constant, not scaled by width)
    - Hidden layers: base_lr / m_d (scaled down as width increases)
    - Output layers: base_lr / m_d (scaled down as width increases)
    - Other (norms): base_lr (constant, no weight decay)

    where m_d = d_model / base_width is the width multiplier.

    Args:
        base_lr: Base learning rate (for embeddings and at base_width)
        d_model: Model hidden dimension
        mup_config: μP configuration
        weight_decay: Weight decay coefficient
        b1, b2: Adam beta parameters
        schedule_fn: Optional learning rate schedule function(step) -> lr

    Returns:
        optax.GradientTransformation optimizer
    """
    if not mup_config.enabled:
        # Standard optimizer without μP scaling
        lr = schedule_fn if schedule_fn else base_lr
        return optax.adamw(lr, b1=b1, b2=b2, weight_decay=weight_decay)

    m_d = compute_width_multiplier(d_model, mup_config.base_width)

    def make_lr(scale: float):
        """Create learning rate (or schedule) with given scale factor."""
        if schedule_fn is not None:
            # Wrap schedule to apply scaling
            def scaled_schedule(count):
                return schedule_fn(count) * scale / base_lr
            return scaled_schedule
        else:
            return base_lr * scale

    # Learning rates for each category
    # Embeddings: constant LR (not scaled by width)
    lr_embedding = make_lr(1.0)
    # Hidden layers: LR / m_d (scales down with width)
    lr_hidden = make_lr(1.0 / m_d)
    # Output layers: LR / m_d (same as hidden)
    lr_output = make_lr(1.0 / m_d)
    # Other (norms, biases): constant LR, no weight decay
    lr_other = make_lr(1.0)

    # Create per-category optimizers
    optimizers = {
        "embedding": optax.adamw(lr_embedding, b1=b1, b2=b2, weight_decay=weight_decay),
        "hidden": optax.adamw(lr_hidden, b1=b1, b2=b2, weight_decay=weight_decay),
        "output": optax.adamw(lr_output, b1=b1, b2=b2, weight_decay=weight_decay),
        "other": optax.adamw(lr_other, b1=b1, b2=b2, weight_decay=0.0),  # No WD for norms
    }

    return optax.multi_transform(optimizers, create_mup_label_fn(mup_config))


def get_mup_init(
    layer_type: str,
    d_model: int,
    mup_config: Optional[MuPConfig],
    mesh_rules: Callable,
) -> nnx.Initializer:
    """
    Get the appropriate initializer for a layer type with μP scaling.

    Args:
        layer_type: One of "hidden", "output", "embedding"
        d_model: Model hidden dimension
        mup_config: μP configuration (None uses standard init)
        mesh_rules: Sharding rules for the initializer

    Returns:
        Wrapped initializer with appropriate scaling
    """
    if mup_config is None or not mup_config.enabled:
        if layer_type == "embedding":
            init = nnx.initializers.normal(stddev=1.0)
        else:
            init = nnx.initializers.lecun_normal()
    else:
        if layer_type == "output":
            init = mup_output_init(d_model, mup_config.base_width)
        elif layer_type == "embedding":
            init = mup_embed_init(stddev=1.0)  # Keep consistent with current
        else:  # hidden
            init = nnx.initializers.lecun_normal()

    return init
