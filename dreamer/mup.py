"""
Complete(d)-Parameterization utilities for Dreamer 4.

Extends μP (Maximal Update Parameterization) with:
- Depth scaling with residual multipliers
- Batch size transfer via SDE-based scaling
- Token horizon transfer for different training durations

References:
- "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
   Hyperparameter Transfer" (Yang et al., 2022)
- "Completed Hyperparameter Transfer Across Modules, Width, Depth,
   Batch & Duration" (arXiv:2512.22382)

Key principles:
- Width scaling: LR/m_N, WD*m_N, eps/m_N for hidden layers
- Depth scaling: residual_mult = m_L^(-α), LR * m_L^(α-1), eps/m_L^α
- Batch/duration: SDE factor √(m_B/m_D) applied to LR and WD
- Attention scaling: 1/d_head (not 1/√d_head)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from dreamer.configs import MuPConfig


# =============================================================================
# Multiplier Computations
# =============================================================================

def compute_width_multiplier(d_model: int, base_width: int) -> float:
    """Compute width multiplier m_N = d_model / base_width."""
    return d_model / base_width


def compute_depth_multiplier(depth: int, base_depth: int) -> float:
    """Compute depth multiplier m_L = depth / base_depth."""
    return depth / base_depth


def compute_batch_multiplier(batch_size: int, base_batch_size: int) -> float:
    """Compute batch multiplier m_B = batch_size / base_batch_size."""
    return batch_size / base_batch_size


def compute_duration_multiplier(total_tokens: int, base_tokens: int) -> float:
    """Compute duration multiplier m_D = total_tokens / base_tokens."""
    return total_tokens / base_tokens


def compute_residual_multiplier(depth: int, mup_config: MuPConfig) -> float:
    """
    Compute residual multiplier for depth scaling.

    residual_mult = m_L^(-α) where m_L = depth / base_depth

    This dampens residual connections in deeper models to maintain
    stable gradient flow.

    Args:
        depth: Actual model depth
        mup_config: Complete(d)P configuration

    Returns:
        Scalar multiplier for residual connections
    """
    if not mup_config.enabled:
        return 1.0
    m_L = compute_depth_multiplier(depth, mup_config.base_depth)
    return m_L ** (-mup_config.depth_alpha)


# =============================================================================
# Initialization Helpers
# =============================================================================

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
    σ = 1 / m_N = base_width / d_model

    This ensures output activations don't scale with width.

    Args:
        d_model: Model hidden dimension
        base_width: Reference width for scaling

    Returns:
        nnx.Initializer with appropriate std
    """
    m_N = compute_width_multiplier(d_model, base_width)
    stddev = 1.0 / m_N
    return nnx.initializers.normal(stddev=stddev)


def mup_embed_init(stddev: float = 0.02) -> nnx.Initializer:
    """
    Create initializer for embeddings with constant std (not scaled by width).

    Embeddings have a single infinite dimension (vocab × d_model), so they
    should use constant initialization std regardless of width.
    """
    return nnx.initializers.normal(stddev=stddev)


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


# =============================================================================
# Parameter Classification
# =============================================================================

def classify_param_for_completedp(path: tuple, leaf: Any) -> str:
    """
    Classify a parameter path into Complete(d)P categories for optimizer scaling.

    Categories:
        - "embedding": Embeddings and learned tokens (special epsilon scaling)
        - "hidden": Hidden layer weights (full width/depth/batch scaling)
        - "output": Output projection weights (same as hidden)
        - "qk_norm": QK normalization scales (inverted SDE factor for epsilon)
        - "norm": Other normalization layers (no width/depth scaling)
        - "other": Biases, etc. (no weight decay)

    Args:
        path: Tuple path to the parameter in the pytree
        leaf: The parameter array (unused, but required by tree_map_with_path)

    Returns:
        Category string for this parameter
    """
    path_str = '.'.join(str(p) for p in path).lower()

    # QK norm scales need special treatment (inverted SDE factor for epsilon)
    # Per the paper, QK norm multipliers share weights across heads
    if ('q_ln' in path_str or 'k_ln' in path_str) and 'scale' in path_str:
        return "qk_norm"

    # Output layers - need LR / m_N and smaller init
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

    # Embeddings - special epsilon scaling (eps/m_N)
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

    # Normalization layers (not QK norm) - no width/depth scaling
    if 'norm' in path_str or 'scale' in path_str:
        return "norm"

    # Default: hidden layers (kernels)
    if 'kernel' in path_str:
        return "hidden"

    # Everything else (biases, etc.)
    return "other"


def create_completedp_label_fn():
    """
    Create a function that labels parameters for Complete(d)P-aware optimization.

    Returns a function compatible with optax.multi_transform.
    """
    def label_fn(params):
        return jax.tree_util.tree_map_with_path(classify_param_for_completedp, params)
    return label_fn


# Legacy alias for backward compatibility
def classify_param_for_mup(path: tuple, leaf: Any) -> str:
    """Alias for classify_param_for_completedp."""
    return classify_param_for_completedp(path, leaf)


def create_mup_label_fn(mup_config: MuPConfig):
    """Legacy alias for create_completedp_label_fn."""
    return create_completedp_label_fn()


# =============================================================================
# Optimizer Factory
# =============================================================================

def create_completedp_optimizer(
    base_lr: float,
    d_model: int,
    depth: int,
    batch_size: int,
    total_tokens: int,
    mup_config: MuPConfig,
    weight_decay: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    schedule_fn: Optional[Callable[[int], float]] = None,
) -> optax.GradientTransformation:
    """
    Create an optimizer with Complete(d)P scaling.

    Complete(d)P scaling rules:

    Width (m_N = d_model / base_width):
        - Hidden/output LR: / m_N
        - Hidden/output WD: * m_N
        - Hidden/output/embedding eps: / m_N

    Depth (m_L = depth / base_depth, α = depth_alpha):
        - Residual mult: m_L^(-α)  [applied in model, not here]
        - Hidden/output LR: * m_L^(α-1)
        - Hidden/output eps: / m_L^α

    Batch/Duration (SDE factor = √(m_B / m_D)):
        - All LRs: * SDE factor
        - All WDs: * SDE factor
        - QK norm eps: * inverted SDE factor (√(m_D / m_B))

    Args:
        base_lr: Base learning rate (tuned at base scale)
        d_model: Model hidden dimension
        depth: Model depth (number of layers)
        batch_size: Actual batch size
        total_tokens: Total training tokens
        mup_config: Complete(d)P configuration
        weight_decay: Base weight decay coefficient
        b1, b2: Adam beta parameters
        eps: Base Adam epsilon
        schedule_fn: Optional learning rate schedule function(step) -> lr

    Returns:
        optax.GradientTransformation optimizer
    """
    if not mup_config.enabled:
        # Standard optimizer without Complete(d)P scaling
        lr = schedule_fn if schedule_fn else base_lr
        return optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)

    # Compute all multipliers
    m_N = compute_width_multiplier(d_model, mup_config.base_width)
    m_L = compute_depth_multiplier(depth, mup_config.base_depth)
    m_B = compute_batch_multiplier(batch_size, mup_config.base_batch_size)
    m_D = compute_duration_multiplier(total_tokens, mup_config.base_tokens)
    alpha = mup_config.depth_alpha

    # SDE-based batch/duration correction factor
    # When batch size increases or training duration decreases, we need larger LR
    sde_factor = (m_B / m_D) ** 0.5

    # -------------------------------------------------------------------------
    # Learning rate scales (relative to base_lr)
    # -------------------------------------------------------------------------
    # Embeddings: no width/depth scaling, but apply SDE factor
    lr_scale_embed = sde_factor

    # Hidden/output: 1/m_N for width, m_L^(α-1) for depth, SDE factor
    lr_scale_hidden = (1.0 / m_N) * (m_L ** (alpha - 1)) * sde_factor

    # Norms: no width/depth scaling, just SDE factor
    lr_scale_norm = sde_factor

    # -------------------------------------------------------------------------
    # Weight decay scales
    # -------------------------------------------------------------------------
    # Hidden/output: m_N * SDE factor (width scaling inverts the LR scaling)
    wd_hidden = weight_decay * m_N * sde_factor

    # Embeddings: just SDE factor (no width scaling)
    wd_embed = weight_decay * sde_factor

    # Norms and others: no weight decay
    wd_other = 0.0

    # -------------------------------------------------------------------------
    # Epsilon scales
    # -------------------------------------------------------------------------
    # Hidden/output: 1/(m_N * m_L^α) * SDE factor
    eps_hidden = eps / m_N / (m_L ** alpha) * sde_factor

    # Embeddings: 1/m_N * SDE factor (paper correction for embedding epsilon)
    eps_embed = eps / m_N * sde_factor

    # QK norm: 1/m_L^α * INVERTED SDE factor (per paper)
    # QK norm weights are shared across heads, requiring special treatment
    eps_qk_norm = eps / (m_L ** alpha) * (m_D / m_B) ** 0.5

    # Other norms: just SDE factor
    eps_norm = eps * sde_factor

    # -------------------------------------------------------------------------
    # Build per-category optimizers
    # -------------------------------------------------------------------------
    def make_lr(scale: float):
        """Create learning rate (or schedule) with given scale factor."""
        if schedule_fn is not None:
            def scaled_schedule(count):
                return schedule_fn(count) * scale / base_lr
            return scaled_schedule
        else:
            return base_lr * scale

    optimizers = {
        "embedding": optax.adamw(
            make_lr(lr_scale_embed), b1=b1, b2=b2, eps=eps_embed, weight_decay=wd_embed
        ),
        "hidden": optax.adamw(
            make_lr(lr_scale_hidden), b1=b1, b2=b2, eps=eps_hidden, weight_decay=wd_hidden
        ),
        "output": optax.adamw(
            make_lr(lr_scale_hidden), b1=b1, b2=b2, eps=eps_hidden, weight_decay=wd_hidden
        ),
        "qk_norm": optax.adamw(
            make_lr(lr_scale_norm), b1=b1, b2=b2, eps=eps_qk_norm, weight_decay=wd_other
        ),
        "norm": optax.adamw(
            make_lr(lr_scale_norm), b1=b1, b2=b2, eps=eps_norm, weight_decay=wd_other
        ),
        "other": optax.adamw(
            make_lr(lr_scale_norm), b1=b1, b2=b2, eps=eps_norm, weight_decay=wd_other
        ),
    }

    return optax.multi_transform(optimizers, create_completedp_label_fn())


# Legacy alias for backward compatibility
def create_mup_optimizer(
    base_lr: float,
    d_model: int,
    mup_config: MuPConfig,
    weight_decay: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.9,
    schedule_fn: Optional[Callable[[int], float]] = None,
    # New optional parameters with defaults for backward compat
    depth: Optional[int] = None,
    batch_size: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> optax.GradientTransformation:
    """
    Legacy API for creating μP optimizer. Use create_completedp_optimizer for full features.

    When depth/batch_size/total_tokens are not provided, uses base values from config.
    """
    # Use base values if not provided (backward compatibility)
    if depth is None:
        depth = mup_config.base_depth
    if batch_size is None:
        batch_size = mup_config.base_batch_size
    if total_tokens is None:
        total_tokens = mup_config.base_tokens

    return create_completedp_optimizer(
        base_lr=base_lr,
        d_model=d_model,
        depth=depth,
        batch_size=batch_size,
        total_tokens=total_tokens,
        mup_config=mup_config,
        weight_decay=weight_decay,
        b1=b1,
        b2=b2,
        schedule_fn=schedule_fn,
    )
