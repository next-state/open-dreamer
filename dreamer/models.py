import einops
import jax.numpy as jnp
from flax import nnx
import jax
from typing import Optional, Tuple, Any, Dict
from einops import rearrange, repeat
import math
from .utils import (
    Modality, TokenLayout,
    normalize_with_dataset_stats,
    unnormalize_with_dataset_stats,
    to_jnp_dtype,
    patchify, unpatchify
)
from .configs import (
    TokenizerModelConfig, DynamicsModelConfig, EncoderModelConfig, DecoderModelConfig,
    TaskEmbedderModelConfig, PolicyHeadModelConfig, RewardHeadModelConfig,
)
from .parallel import MeshRules


# ============================================================================
# KV Cache
# ============================================================================

@jax.tree_util.register_pytree_node_class
class KVCache:
    """
    Ring buffer KV cache for JIT compilation.
    """
    def __init__(self, k, v, index, window_size):
        self.k = k               # (B, Max_T, K, H)
        self.v = v               # (B, Max_T, K, H)
        self.index = index       # scalar integer (i32)
        self.window_size = window_size  # static int

    def tree_flatten(self):
        children = (self.k, self.v, self.index)
        aux_data = (self.window_size,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        k, v, index = children
        window_size, = aux_data
        return cls(k, v, index, window_size)

    @classmethod
    def init(cls, batch_size, window_size, num_kv_heads, head_dim, dtype=jnp.float32):
        dtype = to_jnp_dtype(dtype)
        return cls(
            k=jnp.zeros((batch_size, window_size, num_kv_heads, head_dim), dtype=dtype),
            v=jnp.zeros((batch_size, window_size, num_kv_heads, head_dim), dtype=dtype),
            index=jnp.array(0, dtype=jnp.int32),
            window_size=window_size
        )

    def update(self, k_new, v_new):
        """
        Writes k_new/v_new into the buffer.
        Handles both contiguous writes and wrapping writes (T > 1) via branching.
        Returns updated cache with index incremented by the length of new data.
        """
        T = k_new.shape[1]
        write_idx = self.index % self.window_size

        def _update_contiguous(operand, update, start_idx):
            # Fast path
            return jax.lax.dynamic_update_slice(operand, update, (0, start_idx, 0, 0))

        def _update_wrap(operand, update, start_idx):
            # Slow path
            indices = (jnp.arange(T) + start_idx) % self.window_size
            return operand.at[:, indices, :, :].set(update)

        fits_contiguous = (write_idx + T) <= self.window_size

        k_updated = jax.lax.cond(
            fits_contiguous,
            _update_contiguous,
            _update_wrap,
            self.k, k_new, write_idx
        )
        v_updated = jax.lax.cond(
            fits_contiguous,
            _update_contiguous,
            _update_wrap,
            self.v, v_new, write_idx
        )

        return KVCache(k=k_updated, v=v_updated, index=self.index + T, window_size=self.window_size)

    def get_ordered_kv(self, query_len):
        """
        Returns rolled K, V, and a mask indicating valid data.
        """
        write_idx = self.index % self.window_size
        shift = -1 * write_idx

        # newest is at index -1
        k_ordered = jnp.roll(self.k, shift, axis=1)
        v_ordered = jnp.roll(self.v, shift, axis=1)

        # Valid data is at the END of the buffer: indices [Window-valid_len : Window]
        valid_len = jnp.minimum(self.index, self.window_size)

        # Block garbage data during warmup
        k_idx = jnp.arange(self.window_size)[None, None, None, :] # (1, 1, 1, Window)
        start_valid = self.window_size - valid_len
        valid_mask = k_idx >= start_valid

        # Shifted causal mask
        q_idx = jnp.arange(query_len)[None, None, :, None]
        causal_mask = k_idx <= (self.window_size - query_len + q_idx)

        final_mask = jnp.logical_and(valid_mask, causal_mask)

        return k_ordered, v_ordered, final_mask


def create_transformer_caches(
    depth: int,
    time_every: int,
    flattened_batch_size: int,
    window_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=jnp.float32,
) -> Dict[int, KVCache]:
    """
    Creates KV cache dictionary for transformer layers.

    Args:
        depth: Total number of transformer layers
        time_every: Create cache every N layers (for time attention)
        flattened_batch_size: Batch size after spatial flattening (B * S)
        window_size: Maximum temporal sequence length
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per attention head
        dtype: Data type for cache buffers

    Returns:
        Dictionary mapping time layer indices to KVCache objects
    """
    caches = {}
    for i in range(depth):
        time_index, time_offset = divmod(i, time_every)
        if time_offset == 0:
            caches[time_index] = KVCache.init(
                batch_size=flattened_batch_size,
                window_size=window_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype
            )
    return caches


# ============================================================================
# Building Blocks
# ============================================================================

class RotaryEmbedding1D(nnx.Module):
    """Rotary Position Embedding with precomputed frequencies."""

    def __init__(self, dim: int, theta: float = 10000.0, dtype: Any = jnp.float32, param_dtype: Any = jnp.float32):
        self.dim = dim
        self.theta = theta
        self.dtype = to_jnp_dtype(dtype)
        self.param_dtype = to_jnp_dtype(param_dtype)

        # Precompute inverse frequencies as a Variable (not a Param, so it won't be trained)
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = nnx.Variable(inv_freq)

    def __call__(self, q, k, start_pos=0):
        """
        q: (B, T, N, H)
        k: (B, T, K, H)
        start_pos: scalar int (the absolute position of the first frame in T)
        """
        T = q.shape[1]
        t_indices = jnp.arange(T, dtype=self.dtype) + start_pos
        freqs = jnp.outer(t_indices, self.inv_freq.value) # (T, dim//2)

        freqs_cos = jnp.cos(freqs).astype(self.dtype)
        freqs_sin = jnp.sin(freqs).astype(self.dtype)

        # (1, T, 1, Dim//2)
        freqs_cos = freqs_cos[None, :, None, :]
        freqs_sin = freqs_sin[None, :, None, :]

        return self._apply(q, k, freqs_cos, freqs_sin)

    def _apply(self, xq, xk, freqs_cos, freqs_sin):
        """
        Apply Rotary Positional Embeddings (RoPE) to queries and keys using real sin/cos.
        xq: (B, T, N, H)
        xk: (B, T, K, H)
        freqs_cos: (L, H/2)
        freqs_sin: (L, H/2)
        """
        # Rearrange to (..., H/2, 2)
        xq_pairs = rearrange(xq, '... (d two) -> ... d two', two=2)
        xk_pairs = rearrange(xk, '... (d two) -> ... d two', two=2)

        xq_r, xq_i = xq_pairs[..., 0], xq_pairs[..., 1]
        xk_r, xk_i = xk_pairs[..., 0], xk_pairs[..., 1]

        # Rotation:
        # x' = x cos - y sin
        # y' = x sin + y cos
        xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
        xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

        xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
        xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

        # Stack back and flatten
        xq_out = jnp.stack([xq_out_r, xq_out_i], axis=-1)
        xk_out = jnp.stack([xk_out_r, xk_out_i], axis=-1)

        xq_out = rearrange(xq_out, '... d two -> ... (d two)')
        xk_out = rearrange(xk_out, '... d two -> ... (d two)')

        return xq_out, xk_out


class MAEReplacer(nnx.Module):
    """Masked Autoencoder token replacer for training."""

    def __init__(self, D: int, p_min: float = 0.0, p_max: float = 0.9,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *,
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.p_min = p_min
        self.p_max = p_max
        self.dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        # Learnable mask token
        self.mask_token = nnx.Param(jax.random.normal(rngs.params(), (D,), dtype=param_dtype) * 0.02, sharding_names=mesh_rules('embed'))

    def __call__(self, patches_btnd: jnp.ndarray, *, rngs: nnx.Rngs) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # patches_btnd: (B,T,Np,D)
        B, T, Np, D = patches_btnd.shape
        mask_token = self.mask_token.value.astype(self.dtype)

        # Draw RNGs
        p_rng = rngs.mae()
        m_rng = rngs.mae()
        p_bt = jax.random.uniform(p_rng, (B, T), minval=self.p_min, maxval=self.p_max)  # (B,T)
        keep_prob_bt1 = 1.0 - p_bt[..., None]                                           # (B,T,1)
        keep = jax.random.bernoulli(m_rng, keep_prob_bt1, (B, T, Np))                   # (B,T,Np)
        keep = keep[..., None]                                                          # (B,T,Np,1)
        replaced = jnp.where(keep, patches_btnd, mask_token.reshape(1, 1, 1, D))
        mae_mask = (~keep).astype(jnp.bool_)                                            # (B,T,Np,1)
        return replaced, mae_mask, keep_prob_bt1


class MLP(nnx.Module):
    """Transformer MLP with optional SwiGLU gating."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout_rate: float = 0.0,
                 swiglu: bool = True, parity_2over3: bool = False, use_norm: bool = True,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *, 
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.swiglu = swiglu
        self.parity_2over3 = parity_2over3
        self.use_norm = use_norm
        self.use_bias = use_bias
        self.use_rmsnorm_scale = use_rmsnorm_scale
        self.dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        # Choose hidden size
        mult = self.mlp_ratio
        if self.swiglu and self.parity_2over3:
            mult = self.mlp_ratio * (2.0 / 3.0)  # param parity with GELU MLP

        hidden = int(self.d_model * mult)

        if self.use_norm:
            self.norm = nnx.RMSNorm(d_model, use_scale=self.use_rmsnorm_scale, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

        if self.swiglu:
            self.fc_in = nnx.Linear(d_model, 2 * hidden, use_bias=self.use_bias, dtype=self.dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)
        else:
            self.fc_in = nnx.Linear(d_model, hidden, use_bias=self.use_bias, dtype=self.dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)

        self.fc_out = nnx.Linear(hidden, d_model, use_bias=self.use_bias, dtype=self.dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:
        if self.use_norm:
            x = self.norm(x)

        if self.swiglu:
            # SwiGLU: Dense -> split -> u * silu(v)
            pre = self.fc_in(x)  # (..., 2H)
            u, v = jnp.split(pre, 2, axis=-1)     # (..., H), (..., H)
            h = u * jax.nn.silu(v)                # (..., H)
        else:
            # Standard GELU MLP
            h = self.fc_in(x)
            h = nnx.gelu(h)

        h = self.dropout(h, deterministic=deterministic, rngs=rngs)
        y = self.fc_out(h)
        y = self.dropout(y, deterministic=deterministic, rngs=rngs)
        return y


class GroupedQueryAttention(nnx.Module):
    """Grouped Query Attention with optional QK normalization and RoPE."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 dropout_rate: float = 0.0, qk_norm_type: str | None = None,
                 is_causal: bool = False, rope_theta: float = 10000.0,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *, 
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_rate = dropout_rate
        self.qk_norm_type = qk_norm_type
        self.is_causal = is_causal
        self.rope_theta = rope_theta
        self.use_bias = use_bias
        self.use_rmsnorm_scale = use_rmsnorm_scale
        dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)
        self.dtype = dtype
        
        assert self.dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        self.to_q = nnx.Linear(dim, dim, use_bias=self.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('attn')), rngs=rngs)
        self.to_kv = nnx.Linear(dim, 2 * kv_dim, use_bias=self.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('attn')), rngs=rngs)
        self.to_out = nnx.Linear(dim, dim, use_bias=self.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('attn')), rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate)

        if self.qk_norm_type == 'qknorm':
            self.q_ln = nnx.RMSNorm(head_dim, use_scale=self.use_rmsnorm_scale, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
            self.k_ln = nnx.RMSNorm(head_dim, use_scale=self.use_rmsnorm_scale, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

        self.rope = RotaryEmbedding1D(dim=head_dim, theta=self.rope_theta, dtype=dtype, param_dtype=param_dtype)

    def __call__(self, x, mask, *args, cache: Optional[KVCache] = None, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None):
        """
        https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html
        B = batch size
        S = length of the key/value (source)
        T = length of the query (target)
        N = number of attention heads
        H = dimensions of each attention head
        K = number of key/value heads
        G = number of groups, which equals to N // K
        """
        q = self.to_q(x)
        q = rearrange(q, "B T (N H) -> B T N H", N=self.num_heads)
        kv = self.to_kv(x)
        k, v = rearrange(kv, "B S (C K H) -> C B S K H", C=2, K=self.num_kv_heads)

        scale = q.shape[-1] ** -0.5
        if self.qk_norm_type == 'qknorm':
            q = self.q_ln(q).astype(self.dtype)
            k = self.k_ln(k).astype(self.dtype)
        elif self.qk_norm_type == 'quest':
            # claims to beat qknorm https://openreview.net/pdf?id=HkztQWZfl2
            k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
            scale = 1.0

        # RoPE
        start_pos = cache.index if cache is not None else 0
        q, k = self.rope(q, k, start_pos=start_pos)

        # KV cache
        if self.is_causal and cache is not None:
            # CACHED INFERENCE MODE
            new_cache = cache.update(k, v)

            T = q.shape[1]
            k_attn, v_attn, cache_mask = new_cache.get_ordered_kv(query_len=T)

            attn_is_causal = False # Handled manually by cache_mask
            if mask is not None:
                mask_attn = jnp.logical_and(mask, cache_mask)
            else:
                mask_attn = cache_mask
        else:
            # TRAINING or NON-CAUSAL (SPACE) ATTENTION
            new_cache = None
            k_attn, v_attn = k, v
            mask_attn = mask
            attn_is_causal = self.is_causal

        # SDPA
        attn = jax.nn.dot_product_attention(
            q, k_attn, v_attn,
            mask=mask_attn,
            scale=scale,
            is_causal=attn_is_causal
        )  # TODO: try setting implementation="cudnn"
        attn = rearrange(attn, "B T N H -> B T (N H)")

        out = self.to_out(attn)
        out = self.dropout(out, deterministic=deterministic, rngs=rngs)

        return out, new_cache

class SpaceSelfAttention(nnx.Module):
    """Space self-attention with modality routing."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout_rate: float = 0.0,
                 qk_norm_type: str | None = None, rope_theta: float = 10000.0,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *, 
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.attn = GroupedQueryAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
            rope_theta=rope_theta, is_causal=False,
            use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype, param_dtype=param_dtype, 
            mesh_rules=mesh_rules, rngs=rngs
        )

    def __call__(self, x, mask, *, deterministic: bool = True, cache: Optional[KVCache] = None, rngs: Optional[nnx.Rngs] = None):
        # x: (B, T, S, D)  -> attention across S within each (B,T)
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B T) S D")

        out, _ = self.attn(x, mask=mask, cache=None, deterministic=deterministic, rngs=rngs)

        out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
        return out, None  # Return None for cache consistency

class TimeSelfAttention(nnx.Module):
    """Time self-attention."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout_rate: float = 0.0,
                 qk_norm_type: str | None = None, rope_theta: float = 10000.0,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *, 
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.attn = GroupedQueryAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
            rope_theta=rope_theta, is_causal=True,
            use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype, param_dtype=param_dtype, 
            mesh_rules=mesh_rules, rngs=rngs
        )

    def __call__(self, x, mask, *, deterministic: bool = True, cache: Optional[KVCache] = None, rngs: Optional[nnx.Rngs] = None):
        # mask does nothing, but is required for API consistency
        # x: (B, T, S, D) -> attention across T, causal
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B S) T D")

        out, new_cache = self.attn(x, mask=None, cache=cache, deterministic=deterministic, rngs=rngs)

        out = rearrange(out, "(B S) T D -> B T S D", B=B, S=S)
        return out, new_cache

class BlockCausalLayer(nnx.Module):
    """Single block-causal transformer layer (alternating space/time attention)."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 dropout_rate: float = 0.0, qk_norm_type: str | None = None,
                 mlp_ratio: float = 4.0, layer_index: int = 0, time_every: int = 4,
                 rope_theta: float = 10000.0, use_bias: bool = False, 
                 use_rmsnorm_scale: bool = True, dtype: Any = jnp.float32, 
                 param_dtype: Any = jnp.float32, *, rngs: nnx.Rngs,
                 mesh_rules: MeshRules):
        self.layer_index = layer_index
        self.time_every = time_every
        param_dtype = to_jnp_dtype(param_dtype)

        self.norm = nnx.RMSNorm(dim, use_scale=use_rmsnorm_scale, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

        # Time or space attention
        self.use_time = (self.layer_index + 1) % self.time_every == 0
        if self.use_time:
            self.attn = TimeSelfAttention(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
                rope_theta=rope_theta, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
                dtype=dtype, param_dtype=param_dtype, 
                mesh_rules=mesh_rules, rngs=rngs
            )
        else:
            self.attn = SpaceSelfAttention(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
                rope_theta=rope_theta, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
                dtype=dtype, param_dtype=param_dtype, 
                mesh_rules=mesh_rules, rngs=rngs
            )

        # MLP
        self.mlp = MLP(dim, mlp_ratio, dropout_rate, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale, dtype=dtype, param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs)

    def __call__(self, x, mask, *, deterministic: bool = True, cache: Optional[KVCache] = None, rngs: Optional[nnx.Rngs] = None):
        # Attention (time or space, depending on layer_index)
        y = self.norm(x)
        y, new_cache = self.attn(y, mask=mask, deterministic=deterministic, cache=cache, rngs=rngs)
        x = x + y

        # MLP
        y = self.mlp(x, deterministic=deterministic, rngs=rngs)
        x = x + y
        return x, new_cache

class BlockCausalTransformer(nnx.Module):
    """Stack of block-causal transformer layers."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, depth: int,
                 dropout_rate: float = 0.0, qk_norm_type: str | None = None,
                 mlp_ratio: float = 4.0, time_every: int = 4, rope_theta: float = 10000.0,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32,
                 use_residual_lambdas: bool = False, *,
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.d_model = d_model
        self.depth = depth
        self.time_every = time_every
        self.use_residual_lambdas = use_residual_lambdas
        param_dtype = to_jnp_dtype(param_dtype)

        if use_residual_lambdas:
            self.resid_lambdas = nnx.Param(
                jnp.ones(depth, dtype=param_dtype),
                sharding_names=None
            )
            self.x0_lambdas = nnx.Param(
                jnp.zeros(depth, dtype=param_dtype),
                sharding_names=None
            )

        # Create layers
        self.layers = nnx.List([
            BlockCausalLayer(
                dim=d_model, num_heads=n_heads, num_kv_heads=n_kv_heads,
                dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
                mlp_ratio=mlp_ratio, layer_index=i, time_every=time_every,
                rope_theta=rope_theta, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
                dtype=dtype, param_dtype=param_dtype,
                mesh_rules=mesh_rules, rngs=rngs
            ) for i in range(depth)
        ])

    def __call__(self, x, mask, *, deterministic: bool = True, caches: Optional[Dict[int, KVCache]] = None, rngs: Optional[nnx.Rngs] = None) ->Tuple[jax.Array, KVCache | None]:
        """
        Args:
            x: input tensor
            mask: attention mask
            deterministic: whether to use deterministic mode (no dropout)
            caches: optional dict mapping layer_index -> KVCache
            rngs: optional RNG state for dropout

        Returns:
            output tensor and updated caches (always returns tuple, caches can be None)
        """
        new_caches = {} if caches is not None else None
        x0 = x

        for i, layer in enumerate(self.layers):
            if self.use_residual_lambdas:
                x = self.resid_lambdas.value[i] * x + self.x0_lambdas.value[i] * x0

            time_index = i // self.time_every
            is_time_layer = (i + 1) % self.time_every == 0
            cache_i = caches.get(time_index) if caches is not None and is_time_layer else None

            x, new_cache_i = layer(x, mask=mask, deterministic=deterministic, cache=cache_i, rngs=rngs)

            if new_caches is not None and new_cache_i is not None:
                new_caches[time_index] = new_cache_i

        return x, new_caches

    def estimate_attention_flops(self, batch_size: int, seq_time: int, seq_space: int) -> int:
        """Attention FLOPs per training step (forward + backward).

        Computes FLOPs for Q@K^T and attn@V operations only (not weight matrices).
        Factor of 12 = 2 matmuls × 2 ops × 3 (forward + backward).
        """
        n_time = self.depth // self.time_every
        n_space = self.depth - n_time
        space_attn = 12 * n_space * self.d_model * (seq_space ** 2) * batch_size * seq_time
        time_attn = 12 * n_time * self.d_model * (seq_time ** 2) * batch_size * seq_space
        return int(space_attn + time_attn)

    def count_excluded_params(self) -> int:
        """Count params excluded from FLOP estimation (per-layer scalars)."""
        if not self.use_residual_lambdas:
            return 0
        return self.resid_lambdas.value.size + self.x0_lambdas.value.size


# ============================================================================
# Tokenizer
# ============================================================================

class Encoder(nnx.Module):
    """Vision encoder with MAE masking."""

    def __init__(self, cfg: EncoderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.n_latents = cfg.n_latents
        self.patch_size = cfg.patch_size
        self.dataset_mean = cfg.dataset_mean
        self.dataset_std = cfg.dataset_std
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        self.patch_proj = nnx.Linear(cfg.patch_size * cfg.patch_size * 3, cfg.d_model, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)
        self.bottleneck_proj = nnx.Linear(cfg.d_model, cfg.d_bottleneck, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, depth=cfg.depth,
            dropout_rate=cfg.dropout_rate, qk_norm_type=cfg.qk_norm_type, mlp_ratio=4.0,
            time_every=cfg.time_every, rope_theta=cfg.rope_theta, 
            use_bias=cfg.use_bias, use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            dtype=dtype, param_dtype=param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules, rngs=rngs
        )

        self.mask_and_replace = MAEReplacer(D=cfg.d_model, p_min=cfg.mae_p_min, p_max=cfg.mae_p_max, dtype=dtype, param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs)
        self.latents_enc = nnx.Param(jax.random.normal(rngs.params(), (cfg.n_latents, cfg.d_model), dtype=param_dtype) * 0.02, sharding_names=mesh_rules('embed'))

    def __call__(self, videos, *, deterministic: bool = True, packing_factor = None, rngs: nnx.Rngs = None) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1) takes videos in the [0,255] range
        B, T, H, W, C = videos.shape

        normalized_videos = normalize_with_dataset_stats(videos, mean=self.dataset_mean, std=self.dataset_std)
        patch_tokens = patchify(normalized_videos, patch=self.patch_size)
        proj_patches = self.patch_proj(patch_tokens)  # (B,T,Np,D)

        # 2) MAE mask-and-replace on patch tokens (encoder input only during training)
        if deterministic or rngs is None:
            # Skip MAE masking during inference
            proj_patches_masked = proj_patches
            patch_mask = jnp.zeros((B, T, proj_patches.shape[2], 1), dtype=jnp.bool_)
            keep_prob = jnp.ones((B, T, 1))
        else:
            proj_patches_masked, patch_mask, keep_prob = self.mask_and_replace(proj_patches, rngs=rngs)

        # patch_mask is (B,T,Np,1), need to expand to pixels (B,T,Np, P*P)
        patch_mask_expanded = jnp.repeat(patch_mask, self.patch_size**2, axis=-1)
        frame_mask = unpatchify(patch_mask_expanded, self.patch_size, H, W)

        # 3) Prepend learned latents
        latents = repeat(self.latents_enc.value.astype(proj_patches.dtype), "... -> b t ...", b=B, t=T)
        tokens = jnp.concatenate([latents, proj_patches_masked], axis=2)  # (B,T,S=(Np+Nl),D)

        layout = TokenLayout((
            (Modality.LATENT, self.n_latents),
            (Modality.IMAGE, patch_tokens.shape[-2]),
        ))
        mask = layout.make_mask("encoder")  # (1, 1, q_len, k_len)

        # 5) Feed tokens into transformer
        encoded_tokens, _ = self.transformer(tokens, mask=mask, deterministic=deterministic, rngs=rngs)

        # 6) Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, :self.n_latents]
        proj_tokens = nnx.tanh(self.bottleneck_proj(latent_tokens))

        if packing_factor is not None:
            proj_tokens = rearrange(proj_tokens, "b t (n p) d -> b t n (p d)", p=packing_factor)

        return proj_tokens, (frame_mask, keep_prob)


class Decoder(nnx.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.
    """

    def __init__(self, cfg: DecoderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.n_latents = cfg.n_latents
        self.patch_size = cfg.patch_size
        self.H = cfg.H
        self.W = cfg.W
        self.dataset_mean = cfg.dataset_mean
        self.dataset_std = cfg.dataset_std
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        self.n_patches = (self.H // self.patch_size) * (self.W // self.patch_size)
        self.up_proj = nnx.Linear(cfg.d_bottleneck, cfg.d_model, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)
        self.patch_head = nnx.Linear(cfg.d_model, cfg.d_patch, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')), rngs=rngs)

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, depth=cfg.depth,
            dropout_rate=cfg.dropout_rate, qk_norm_type=cfg.qk_norm_type, mlp_ratio=4.0,
            time_every=cfg.time_every, rope_theta=cfg.rope_theta, 
            use_bias=cfg.use_bias, use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            dtype=dtype, param_dtype=param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules, rngs=rngs
        )

        self.patch_queries = nnx.Param(jax.random.normal(rngs.params(), (self.n_patches, cfg.d_model), dtype=param_dtype) * 0.02, sharding_names=mesh_rules('embed'))

    def get_token_layout(self) -> TokenLayout:
        return TokenLayout((
            (Modality.LATENT, self.n_latents),
            (Modality.IMAGE, self.n_patches)
        ))

    def __call__(self, z: jnp.ndarray, *, deterministic: bool = True, packing_factor = None, caches: Optional[Dict[int, KVCache]] = None, rngs: Optional[nnx.Rngs] = None):
        if packing_factor is not None:
            z = rearrange(z, "... n (p d) -> ... (n p) d", p=packing_factor)

        B, T, N_l, d_bottleneck = z.shape
        # 1) Up-project latent bottleneck to d_model (per latent token)
        latents = self.up_proj(z)  # (B, T, N_l, D)

        # 2) Learned per-patch query tokens (owned by the decoder)
        patches = repeat(self.patch_queries.value.astype(latents.dtype), " ... -> b t ...", b=B, t=T)  # (B, T, Np, D)

        # 3) Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=-2)

        # 5) Make mask
        layout = self.get_token_layout()
        mask = layout.make_mask("decoder")

        x, new_caches = self.transformer(tokens, mask=mask, deterministic=deterministic, caches=caches, rngs=rngs)
        # 6) Prediction head over the patch-query slice

        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = self.patch_head(x_patches)  # (B, T, Np, D_patch)
        out_normalized_frames = unpatchify(pred_btnd, patch=self.patch_size, H=self.H, W=self.W)
        out_frames = unnormalize_with_dataset_stats(out_normalized_frames, mean=self.dataset_mean, std=self.dataset_std)
        return out_frames, new_caches

class Tokenizer(nnx.Module):
    """Complete tokenizer (encoder + decoder)."""

    def __init__(self, cfg: TokenizerModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg

        # Create encoder and decoder
        self.encoder = Encoder(cfg.encoder, mesh_rules=mesh_rules, rngs=rngs)
        self.decoder = Decoder(cfg.decoder, mesh_rules=mesh_rules, rngs=rngs)

    def __call__(self, videos, *, deterministic: bool = True, rngs: nnx.Rngs = None):
        z, aux = self.encoder(videos, deterministic=deterministic, rngs=rngs)
        recon, _ = self.decoder(z, deterministic=deterministic, rngs=rngs)
        return recon, aux

    def encode(self, videos, *, deterministic: bool = True, packing_factor = None, rngs: nnx.Rngs = None):
        return self.encoder(videos, deterministic=deterministic, packing_factor=packing_factor, rngs=rngs)

    def decode(self, z, *, deterministic: bool = True, caches=None, packing_factor = None, rngs: Optional[nnx.Rngs] = None):
        frames, caches = self.decoder(z, deterministic=deterministic, packing_factor=packing_factor, caches=caches, rngs=rngs)
        return frames, caches

    def create_static_caches(self, batch_size: int, window_size: int = 1024, dtype=jnp.float32) -> Dict[int, KVCache]:
        """Creates concrete, zero-filled KV cache buffers for JIT compilation."""
        layout = self.decoder.get_token_layout()

        return create_transformer_caches(
            depth=self.cfg.decoder.depth,
            time_every=self.cfg.decoder.time_every,
            flattened_batch_size=batch_size * layout.S,
            window_size=window_size,
            num_kv_heads=self.cfg.decoder.n_kv_heads,
            head_dim=self.cfg.decoder.d_model // self.cfg.decoder.n_heads,
            dtype=dtype
        )

    def num_scaling_params(self) -> int:
        """Total params for scaling law analysis (Chinchilla-style, includes all)."""
        _, state, _ = nnx.split(self, nnx.Param, ...)
        sizes = jax.tree.map(jnp.size, state)
        return jax.tree.reduce(lambda a, b: a + b, sizes)

    def count_excluded_params(self) -> int:
        """Params to exclude from FLOP estimation (embeddings + scalars)."""
        excluded = 0
        # Learned tokens (embedding-like)
        excluded += self.encoder.latents_enc.value.size
        excluded += self.encoder.mask_and_replace.mask_token.value.size
        excluded += self.decoder.patch_queries.value.size
        # Per-layer scalars
        excluded += self.encoder.transformer.count_excluded_params()
        excluded += self.decoder.transformer.count_excluded_params()
        return excluded

    def estimate_flops(self, batch_size: int, seq_length: int) -> int:
        """FLOPs per training step (forward + backward).

        Uses Karpathy/Bahdanau methodology:
        - 6 FLOPs per weight param per token (excluding embeddings/scalars)
        - Plus attention computation FLOPs (Q@K^T and attn@V)
        """
        S = self.decoder.get_token_layout().S

        # Weight FLOPs (excluding embeddings/scalars)
        total_params = self.num_scaling_params()
        excluded = self.count_excluded_params()
        weight_flops = 6 * (total_params - excluded) * batch_size * seq_length * S

        # Attention FLOPs
        enc_attn = self.encoder.transformer.estimate_attention_flops(batch_size, seq_length, S)
        dec_attn = self.decoder.transformer.estimate_attention_flops(batch_size, seq_length, S)

        return int(weight_flops + enc_attn + dec_attn)


# ============================================================================
# Dynamics
# ============================================================================

class ActionEncoder(nnx.Module):
    """Action encoder for dynamics model."""

    def __init__(self, d_model: int, action_dim: int = 16, dtype: Any = jnp.float32,
                 param_dtype: Any = jnp.float32, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.d_model = d_model
        self.action_dim = action_dim
        dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        # Base "action token" embedding (used always)
        self.base_action_emb = nnx.Param(jax.random.normal(rngs.params(), (d_model,), dtype=param_dtype) * 0.02, sharding_names=mesh_rules('embed'))
        # Embed categorical actions
        self.emb_key = nnx.Embed(
            action_dim, d_model,
            dtype=dtype, param_dtype=param_dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=1.0), mesh_rules('embed')),
            rngs=rngs
        )

    def __call__(self, actions: Optional[jnp.ndarray], batch_time_shape: Optional[Tuple[int,int]] = None, as_tokens: bool = True):
        base_emb = self.base_action_emb.value.astype(self.emb_key.embedding.value.dtype)

        if actions is None:
            # unlabeled videos: just broadcast base embedding
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = jnp.broadcast_to(base_emb, (B, T, self.d_model))
        else:
            # embed categorical actions
            emb_key = self.emb_key(actions)
            out = emb_key + base_emb  # broadcast add

        if as_tokens:
            # expand a token axis (S_a = 1)
            out = out[..., None, :]

        return out


class Dynamics(nnx.Module):
    """Dynamics model (world model)."""

    def __init__(self, cfg: DynamicsModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg
        self.dtype = to_jnp_dtype(cfg.dtype)
        self.param_dtype = to_jnp_dtype(cfg.param_dtype)
        self.d_model = cfg.d_model
        self.d_bottleneck = cfg.d_bottleneck
        self.depth = cfg.depth
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.packing_factor = cfg.packing_factor
        self.n_register = cfg.n_register
        self.k_max = cfg.k_max

        # Project spatial tokens
        self.spatial_proj = nnx.Linear(cfg.d_bottleneck * cfg.packing_factor, cfg.d_model, use_bias=cfg.use_bias, dtype=self.dtype, param_dtype=self.param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)

        # Register tokens
        self.register_tokens = nnx.Param(jax.random.normal(rngs.params(), (cfg.n_register, cfg.d_model), dtype=self.param_dtype) * 0.02, sharding_names=mesh_rules('embed'))

        # Action encoder
        self.action_encoder = ActionEncoder(d_model=cfg.d_model, action_dim=cfg.action_dim, dtype=self.dtype, param_dtype=self.param_dtype, mesh_rules=mesh_rules, rngs=rngs)

        # Transformer
        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            depth=cfg.depth, dropout_rate=cfg.dropout_rate, qk_norm_type=cfg.qk_norm_type,
            mlp_ratio=cfg.mlp_ratio, time_every=cfg.time_every, rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias, use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            dtype=self.dtype, param_dtype=self.param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules, rngs=rngs
        )

        # Discrete embeddings for shortcut conditioning
        # Step size d ∈ {1, 1/2, 1/4, ..., 1/k_max}
        # We index steps by: step_idx = log2(1/d) ∈ {0, 1, 2, ..., log2(k_max)}
        self.num_step_bins = int(math.log2(cfg.k_max)) + 1
        self.step_embed = nnx.Embed(
            self.num_step_bins, cfg.d_model,
            dtype=self.dtype, param_dtype=self.param_dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=1.0), mesh_rules('embed')),
            rngs=rngs
        )

        # Signal level τ ∈ {0, d, 2d, ..., 1 - d, 1}
        # We index signals by: signal_idx = τ * k_max ∈ {0, 1, 2, ..., k_max}
        self.signal_embed = nnx.Embed(
            cfg.k_max, cfg.d_model,
            dtype=self.dtype, param_dtype=self.param_dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=1.0), mesh_rules('embed')),
            rngs=rngs
        )

        # Output head (zero-init)
        self.flow_x_head = nnx.Linear(
            cfg.d_model, cfg.d_bottleneck * cfg.packing_factor,
            use_bias=cfg.use_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
            bias_init=nnx.initializers.zeros,
            dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
        )

    def get_token_layout(self, n_spatial: int, n_agent: int = 0) -> TokenLayout:
        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1),
            (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, self.cfg.n_register),
        ]
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))
        return TokenLayout(tuple(segments))

    def create_static_caches(self, batch_size: int, n_spatial: int, window_size: int = 1024,
                           n_agent: int = 0, dtype=jnp.float32) -> Dict[int, KVCache]:
        """Creates concrete, zero-filled buffers for JIT compilation."""
        layout = self.get_token_layout(n_spatial=n_spatial, n_agent=n_agent)

        return create_transformer_caches(
            depth=self.cfg.depth,
            time_every=self.cfg.time_every,
            flattened_batch_size=batch_size * layout.S,
            window_size=window_size,
            num_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            dtype=dtype
        )

    def __call__(self, actions, step_indices, tau_indices, packed_enc_tokens, *,
                task_embeddings: Optional[jnp.ndarray] = None, deterministic: bool = True,
                caches: Optional[KVCache | None] = None, rngs: Optional[nnx.Rngs] = None
    )->Tuple[jax.Array, Tuple[jax.Array|None, KVCache|None]]:
        """
        Args:
          packed_enc_tokens:      (B, T, n_spatial, d_spatial) packed encoder tokens
          actions:       (B, T) int32 in [0, n_keyboard) raw action tokens
          step_indices:  (B, T) int32 — step indices for embedding lookup
          tau_indices:   (B, T) int32 - signal indices for embedding lookup
          caches:     optional dict of KVCache for each layer

        Shapes produced:
          spatial_tokens: (B, T, n_spatial, d_model)
          action_tokens:  (B, T, 1, d_model)  # if your ActionEncoder emits one token
          signal_token:   (B, T, 1, d_model)
          step_token:     (B, T, 1, d_model)
        """
        # --- 1) Project spatial tokens to model dimension
        spatial_tokens = self.spatial_proj(packed_enc_tokens) # (B, T, n_spatial, d_model)

        # --- 2) Encode actions to d_model
        action_tokens = self.action_encoder(actions)  # (B, T, N_a, d_model)

        # --- 3) Prepare learned register tokens
        B, T = spatial_tokens.shape[:2]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value.astype(self.dtype)[None, None, ...],  # (1,1,n_register,d_model)
            (B, T, self.n_register, self.d_model),
        )

        # --- 4) Shortcut embeddings (discrete lookup)
        step_tok   = self.step_embed(step_indices)[:, :, None, :]         # (B, T, 1, d_model)
        signal_tok = self.signal_embed(tau_indices)[:, :, None, :]     # (B, T, 1, d_model)

        # --- 5) Concatenate in your declared layout order
        if task_embeddings is not None:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, task_embeddings]
        else:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens]
        tokens = jnp.concatenate(toks, axis=2)                    # (B,T,S,D)

        # make the layout for masking
        n_agent = task_embeddings.shape[2] if task_embeddings is not None else 0
        layout = self.get_token_layout(n_spatial=spatial_tokens.shape[2], n_agent=n_agent)
        mask = layout.make_mask("wm_agent")

        x, new_caches = self.transformer(tokens, mask, deterministic=deterministic, caches=caches, rngs=rngs)

        spatial_tokens = x[:, :, layout.slices()[Modality.SPATIAL], :]
        x1_hat = self.flow_x_head(spatial_tokens)
        h_t = x[:, :, layout.slices()[Modality.AGENT], :] if task_embeddings is not None else None  # (B,T,n_agent,D) or None
        return x1_hat, (h_t, new_caches)

    def num_scaling_params(self) -> int:
        """Total params for scaling law analysis (Chinchilla-style, includes all)."""
        _, state, _ = nnx.split(self, nnx.Param, ...)
        sizes = jax.tree.map(jnp.size, state)
        return jax.tree.reduce(lambda a, b: a + b, sizes)

    def count_excluded_params(self) -> int:
        """Params to exclude from FLOP estimation (embeddings + scalars)."""
        excluded = 0
        # Register tokens
        excluded += self.register_tokens.value.size
        # Action encoder embeddings
        excluded += self.action_encoder.base_action_emb.value.size
        excluded += self.action_encoder.emb_key.embedding.value.size
        # Shortcut embeddings
        excluded += self.step_embed.embedding.value.size
        excluded += self.signal_embed.embedding.value.size
        # Per-layer scalars
        excluded += self.transformer.count_excluded_params()
        return excluded

    def estimate_flops(self, batch_size: int, seq_length: int, n_spatial: int) -> int:
        """FLOPs per training step (forward + backward).

        Uses Karpathy/Bahdanau methodology:
        - 6 FLOPs per weight param per token (excluding embeddings/scalars)
        - Plus attention computation FLOPs (Q@K^T and attn@V)
        """
        S = self.get_token_layout(n_spatial=n_spatial).S

        # Weight FLOPs (excluding embeddings/scalars)
        total_params = self.num_scaling_params()
        excluded = self.count_excluded_params()
        weight_flops = 6 * (total_params - excluded) * batch_size * seq_length * S

        # Attention FLOPs
        attn_flops = self.transformer.estimate_attention_flops(batch_size, seq_length, S)

        return int(weight_flops + attn_flops)


class TaskEmbedder(nnx.Module):
    """Task embedder for agent conditioning."""

    def __init__(self, cfg: TaskEmbedderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        if cfg.use_ids:
            self.emb = nnx.Embed(cfg.n_tasks, cfg.d_model, dtype=dtype, param_dtype=param_dtype, embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=1.0), mesh_rules('embed')), rngs=rngs)
        else:
            self.emb = nnx.Linear(cfg.d_task, cfg.d_model, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')), rngs=rngs)

        self.agent_base = nnx.Param(jax.random.normal(rngs.params(), (cfg.d_model,), dtype=param_dtype) * 0.02, sharding_names=mesh_rules('embed'))

    def __call__(self, task, B: int, T: int) -> jax.Array:
        """
        If use_ids=True:
            task: (B,) int32 ids in [0, n_tasks)
        Else:
            task: (B, d_task) float32 vector

        Returns agent tokens: (B, T, n_agent, d_model)
        """
        emb = self.emb(task)
        base = self.agent_base.value.astype(emb.dtype)
        x = emb + base[None, :]

        # Replicate across time and agent slots
        x = jnp.broadcast_to(x[:, None, None, :], (B, T, self.cfg.n_agent, self.cfg.d_model))
        return x


class PolicyHeadMTP(nnx.Module):
    """Multi-Token action prediction."""

    def __init__(self, cfg: PolicyHeadModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        # Feature projector: operates on flattened agent tokens (L*d_model)
        d_flat = cfg.d_model * cfg.L
        self.projector = MLP(
            d_model=d_flat, mlp_ratio=cfg.mlp_ratio, dropout_rate=cfg.dropout_rate,
            swiglu=cfg.swiglu, parity_2over3=cfg.parity_2over3, use_bias=cfg.use_bias, dtype=dtype,
            param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs
        )

        # Single matmul that produces all L offsets at once: (… , d_flat) -> (…, L, A)
        self.out = nnx.Linear(d_flat, cfg.L * cfg.action_dim, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')), rngs=rngs)

    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None) -> jnp.ndarray:
        h_t = einops.rearrange(h_t, 'b t n c -> b t (n c)')
        x = self.projector(h_t, deterministic=deterministic, rngs=rngs)  # (B, T, D)
        logits = self.out(x)                                  # (B, T, L*A)
        logits = rearrange(logits, 'b t (l a) -> b t l a', l=self.cfg.L, a=self.cfg.action_dim)
        return logits


class RewardHeadMTP(nnx.Module):
    """Multi-Token reward prediction with symexp twohot bins."""

    def __init__(self, cfg: RewardHeadModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        d_flat = cfg.d_model * cfg.L
        self.projector = MLP(
            d_model=d_flat, mlp_ratio=cfg.mlp_ratio, dropout_rate=cfg.dropout_rate,
            swiglu=cfg.swiglu, parity_2over3=cfg.parity_2over3, use_bias=cfg.use_bias, dtype=dtype,
            param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs
        )
        self.out = nnx.Linear(d_flat, cfg.L * cfg.num_bins, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')), rngs=rngs)

        # Precompute bin centers as a constant
        self.symexp_centers_log = jnp.linspace(cfg.log_low, cfg.log_high, cfg.num_bins)

    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_t = einops.rearrange(h_t, '... n c -> ... (n c)')
        x = self.projector(h_t, deterministic=deterministic, rngs=rngs)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, L*K)
        logits = rearrange(logits, '... (l k) -> ... l k', l=self.cfg.L, k=self.cfg.num_bins)
        return logits, self.symexp_centers_log


class ValueHead(nnx.Module):
    """Value prediction with symexp twohot bins."""

    def __init__(self, d_model: int, num_bins: int = 101, L: int = 1, mlp_ratio: float = 2.0,
                 dropout_rate: float = 0.0, swiglu: bool = True, parity_2over3: bool = False,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32,
                 log_low: float = -8.0, log_high: float = 8.0, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.d_model = d_model
        self.L = L
        self.num_bins = num_bins
        self.log_low = log_low
        self.log_high = log_high
        dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        d_flat = d_model * L
        self.projector = MLP(
            d_model=d_flat, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate,
            swiglu=swiglu, parity_2over3=parity_2over3, use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale, dtype=dtype,
            param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs
        )
        self.out = nnx.Linear(d_flat, num_bins, use_bias=use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')), rngs=rngs)

        # Precompute bin centers as a constant
        self.symexp_centers_log = jnp.linspace(self.log_low, self.log_high, self.num_bins)


    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True, rngs: Optional[nnx.Rngs] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_t = einops.rearrange(h_t, 'b t n c -> b t (n c)')
        x = self.projector(h_t, deterministic=deterministic, rngs=rngs)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, K)
        return logits, self.symexp_centers_log
