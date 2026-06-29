import einops
import jax.numpy as jnp
from flax import nnx
import jax
from typing import Tuple, Any, Dict, Sequence
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
from .actions import Actions


def _nnx_list(items):
    if hasattr(nnx, "List"):
        return nnx.List(items)
    return list(items)


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


KVCachesDict = Dict[int, KVCache]  # Type alias for transformer KV cache dictionaries
TokenizerKVCachesDict = Dict[str, KVCachesDict | None]  # "encoder" / "decoder" cache namespaces


def create_transformer_caches(
    layer_is_time: Sequence[bool],
    flattened_batch_size: int,
    window_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=jnp.float32,
) -> KVCachesDict:
    """
    Creates KV cache dictionary for transformer layers.

    Args:
        layer_is_time: Boolean flags indicating which layers use temporal attention
        flattened_batch_size: Batch size after spatial flattening (B * S)
        window_size: Maximum temporal sequence length
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per attention head
        dtype: Data type for cache buffers

    Returns:
        Dictionary mapping time layer indices to KVCache objects
    """
    caches = {}
    time_index = 0
    for is_time_layer in layer_is_time:
        if is_time_layer:
            caches[time_index] = KVCache.init(
                batch_size=flattened_batch_size,
                window_size=window_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype
            )
            time_index += 1
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

    def __call__(self, patches_btnd: jnp.ndarray, *, rngs: nnx.Rngs, p_max_override: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # patches_btnd: (B,T,Np,D)
        B, T, Np, D = patches_btnd.shape
        mask_token = self.mask_token.value.astype(self.dtype)

        p_max = p_max_override if p_max_override is not None else self.p_max

        # Draw RNGs
        p_rng = rngs.mae()
        m_rng = rngs.mae()
        p_bt = jax.random.uniform(p_rng, (B, T), minval=self.p_min, maxval=p_max)  # (B,T)
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

    def __call__(self, x: jnp.ndarray,  deterministic: bool = True, rngs: nnx.Rngs | None = None) -> jnp.ndarray:
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
                 use_seq_parallel: bool = False,
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
        self.use_seq_parallel = use_seq_parallel
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

    def __call__(
            self,
            x,
            mask: jnp.ndarray | None = None,
            local_window_size: int | tuple[int, int] | None = None,
            deterministic: bool = True,
            cache: KVCache | None = None,
            rngs: nnx.Rngs | None = None,
            return_weights: bool = False,
        ):
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

        # Sequence parallel path: all-gather K,V across sequence axis for training
        if self.use_seq_parallel and cache is None:
            # Get sequence axis info from the mesh
            seq_axis_size = jax.lax.psum(1, axis_name='seq')
            seq_axis_idx = jax.lax.axis_index('seq')

            T_local = k.shape[1]
            T_global = T_local * seq_axis_size

            # Global position offset for RoPE
            global_start_pos = seq_axis_idx * T_local
            q, k = self.rope(q, k, start_pos=global_start_pos)

            # All-gather K, V across sequence axis
            # tiled=True means the results are concatenated along the axis rather than stacked
            k_full = jax.lax.all_gather(k, axis_name='seq', axis=1, tiled=True)
            v_full = jax.lax.all_gather(v, axis_name='seq', axis=1, tiled=True)

            # Build causal mask for local Q vs full K
            # q_pos: positions of local queries in global sequence
            # k_pos: positions of all keys (0 to T_global-1)
            q_pos = jnp.arange(T_local) + global_start_pos
            k_pos = jnp.arange(T_global)
            causal_mask = q_pos[:, None] >= k_pos[None, :]  # (T_local, T_global)
            causal_mask = causal_mask[None, None, :, :]  # (1, 1, T_local, T_global)

            # Combine with any input mask if provided
            if mask is not None:
                causal_mask = jnp.logical_and(mask, causal_mask)

            # SDPA with explicit causal mask (is_causal=False since we handle it)
            attn = jax.nn.dot_product_attention(
                q, k_full, v_full,
                mask=causal_mask,
                scale=scale,
                is_causal=False,  # Handled by our custom mask
            )
            new_cache = None

            if return_weights:
                G = self.num_heads // self.num_kv_heads
                k_exp = jnp.repeat(k_full, G, axis=2)  # (B, S, N, H)
                logits = jnp.einsum('B T N H, B S N H -> B N T S', q, k_exp) * scale
                logits = jnp.where(causal_mask, logits, jnp.finfo(logits.dtype).min)
                attn_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
            else:
                attn_weights = None

        else:
            # Standard non-SP path
            # RoPE
            start_pos = cache.index if cache is not None else 0
            q, k = self.rope(q, k, start_pos=start_pos)

            # KV cache
            if self.is_causal and cache is not None:
                # CACHED INFERENCE MODE
                new_cache = cache.update(k, v)

                T = q.shape[1]
                k_attn, v_attn, cache_mask = new_cache.get_ordered_kv(query_len=T)

                attn_is_causal = False  # Handled manually by cache_mask
                if mask is not None:
                    mask_attn = jnp.logical_and(mask, cache_mask)
                else:
                    mask_attn = cache_mask
            else:
                # TRAINING or NON-CAUSAL (SPACE) ATTENTION
                new_cache = None
                k_attn, v_attn = k, v
                mask_attn = mask
                attn_is_causal = self.is_causal and (mask is None)

            # SDPA
            attn = jax.nn.dot_product_attention(
                q, k_attn, v_attn,
                mask=mask_attn,
                scale=scale,
                is_causal=attn_is_causal,
                local_window_size=None
            )  # TODO: try setting implementation="cudnn"

            if return_weights:
                G = self.num_heads // self.num_kv_heads
                k_exp = jnp.repeat(k_attn, G, axis=2)  # (B, S, N, H)
                logits = jnp.einsum('B T N H, B S N H -> B N T S', q, k_exp) * scale
                if mask_attn is not None:
                    logits = jnp.where(mask_attn, logits, jnp.finfo(logits.dtype).min)
                elif attn_is_causal:
                    T_q, T_k = q.shape[1], k_attn.shape[1]
                    causal = jnp.tril(jnp.ones((T_q, T_k), dtype=jnp.bool_))
                    logits = jnp.where(causal[None, None], logits, jnp.finfo(logits.dtype).min)
                attn_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
            else:
                attn_weights = None

        attn = rearrange(attn, "B T N H -> B T (N H)")

        out = self.to_out(attn)
        out = self.dropout(out, deterministic=deterministic, rngs=rngs)

        return out, new_cache, attn_weights

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

    def __call__(
            self,
            x,
            *,
            mask: jnp.ndarray | None = None,
            local_window_size: int | tuple[int, int] | None = None,
            deterministic: bool = True,
            cache: KVCache | None = None,
            rngs: nnx.Rngs | None = None,
            return_weights: bool = False,
        ):
        # x: (B, T, S, D)  -> attention across S within each (B,T)
        # Note: local_window_size is ignored for space attention (just for compatibility)
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B T) S D")

        out, _, attn_weights = nnx.remat(self.attn, static_argnums=(2, 3, 4, 6),)(x, mask, None, deterministic, None, rngs, return_weights)
        out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights, "(B T) N S1 S2 -> B T N S1 S2", B=B, T=T)
        return out, None, attn_weights

class TimeSelfAttention(nnx.Module):
    """Time self-attention."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout_rate: float = 0.0,
                 qk_norm_type: str | None = None, rope_theta: float = 10000.0,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 use_seq_parallel: bool = False,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *,
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.attn = GroupedQueryAttention(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
            rope_theta=rope_theta, is_causal=True,
            use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
            use_seq_parallel=use_seq_parallel,
            dtype=dtype, param_dtype=param_dtype,
            mesh_rules=mesh_rules, rngs=rngs
        )

    def __call__(
        self,
        x,
        *,
        mask: jnp.ndarray | None = None,
        local_window_size: int | tuple[int, int] | None = None,
        deterministic: bool = True,
        cache: KVCache | None = None,
        rngs: nnx.Rngs | None = None,
        return_weights: bool = False,
    ):
        # x: (B, T, S, D) -> attention across T, causal
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B S) T D")

        if mask is not None and mask.ndim >= 3 and mask.shape[0] == B:
            mask = repeat(mask, 'B ... -> (B S) ...', S=S)

        out, new_cache, attn_weights = nnx.remat(self.attn, static_argnums=(2, 3, 6),)(x, mask, local_window_size, deterministic, cache, rngs, return_weights)
        out = rearrange(out, "(B S) T D -> B T S D", B=B, S=S)
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights, "(B S) N T1 T2 -> B S N T1 T2", B=B, S=S)
        return out, new_cache, attn_weights

class BlockCausalLayer(nnx.Module):
    """Single block-causal transformer layer (alternating space/time attention)."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 dropout_rate: float = 0.0, qk_norm_type: str | None = None,
                 mlp_ratio: float = 4.0, layer_index: int = 0, time_every: int = 4, time_layer_offset: int = 1,
                 rope_theta: float = 10000.0, use_bias: bool = False,
                 use_rmsnorm_scale: bool = True, use_seq_parallel: bool = False,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32, *,
                 rngs: nnx.Rngs, mesh_rules: MeshRules):
        self.layer_index = layer_index
        self.time_every = time_every
        self.time_layer_offset = time_layer_offset
        param_dtype = to_jnp_dtype(param_dtype)

        self.norm = nnx.RMSNorm(dim, use_scale=use_rmsnorm_scale, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

        # Time or space attention
        is_time_layer = (self.layer_index + self.time_layer_offset) % self.time_every == 0
        if is_time_layer:
            self.attn = TimeSelfAttention(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
                rope_theta=rope_theta, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
                use_seq_parallel=use_seq_parallel,
                dtype=dtype, param_dtype=param_dtype,
                mesh_rules=mesh_rules, rngs=rngs
            )
        else:
            # SpaceSelfAttention doesn't need seq_parallel - it operates on (B*T, S, D)
            # with T already in the batch dimension
            self.attn = SpaceSelfAttention(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
                rope_theta=rope_theta, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
                dtype=dtype, param_dtype=param_dtype,
                mesh_rules=mesh_rules, rngs=rngs
            )

        # MLP
        self.mlp = MLP(dim, mlp_ratio, dropout_rate, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale, dtype=dtype, param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs)

    @property
    def is_time_layer(self) -> bool:
        return isinstance(self.attn, TimeSelfAttention)

    def __call__(
        self,
        x,
        *,
        space_mask: jnp.ndarray | None = None,
        time_mask: jnp.ndarray | None = None,
        time_local_window_size: int | tuple[int, int] | None = None,
        deterministic: bool = True,
        cache: KVCache | None = None,
        rngs: nnx.Rngs | None = None,
        return_weights: bool = False,
    ):
        # Attention (time or space, depending on layer_index)
        y = self.norm(x)
        if self.is_time_layer:
            attn_mask = time_mask
            local_window_size = time_local_window_size
        else:
            attn_mask = space_mask
            local_window_size = None
        y, new_cache, attn_weights = self.attn(y, mask=attn_mask, local_window_size=local_window_size, deterministic=deterministic, cache=cache, rngs=rngs, return_weights=return_weights)
        x = x + y

        # MLP
        y = nnx.remat(self.mlp)(x, deterministic=deterministic, rngs=rngs)
        x = x + y
        return x, new_cache, attn_weights

class BlockCausalTransformer(nnx.Module):
    """Stack of block-causal transformer layers."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, depth: int,
                 dropout_rate: float = 0.0, qk_norm_type: str | None = None,
                 mlp_ratio: float = 4.0, time_every: int = 4, time_layer_offset: int = 1, rope_theta: float = 10000.0,
                 use_bias: bool = False, use_rmsnorm_scale: bool = True,
                 use_seq_parallel: bool = False,
                 dtype: Any = jnp.float32, param_dtype: Any = jnp.float32,
                 use_residual_lambdas: bool = False, *,
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.d_model = d_model
        self.depth = depth
        self.time_every = time_every
        self.time_layer_offset = time_layer_offset
        self.use_residual_lambdas = use_residual_lambdas
        self.dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        if use_residual_lambdas:
            self.resid_lambdas = nnx.Param(
                jnp.ones(depth, dtype=param_dtype),
                sharding_names=(None,)  # Small per-layer params, no sharding needed
            )
            self.x0_lambdas = nnx.Param(
                jnp.zeros(depth, dtype=param_dtype),
                sharding_names=(None,)  # Small per-layer params, no sharding needed
            )

        # Create layers
        self.layers = _nnx_list([
            BlockCausalLayer(
                dim=d_model, num_heads=n_heads, num_kv_heads=n_kv_heads,
                dropout_rate=dropout_rate, qk_norm_type=qk_norm_type,
                mlp_ratio=mlp_ratio, layer_index=i, time_every=time_every, time_layer_offset=time_layer_offset,
                rope_theta=rope_theta, use_bias=use_bias, use_rmsnorm_scale=use_rmsnorm_scale,
                use_seq_parallel=use_seq_parallel,
                dtype=dtype, param_dtype=param_dtype,
                mesh_rules=mesh_rules, rngs=rngs
            ) for i in range(depth)
        ])

    def __call__(
        self,
        x,
        *,
        space_mask: jnp.ndarray | None = None,
        time_mask: jnp.ndarray | None = None,
        time_local_window_size: int | tuple[int, int] | None = None,
        deterministic: bool = True,
        caches: KVCachesDict | None = None,
        rngs: nnx.Rngs | None = None,
        return_weights: bool = False,
    ) -> Tuple[jax.Array, KVCachesDict | None, list | None]:
        """
        Args:
            x: (B, T, S, D) input tensor
            space_mask: optional spatial attention mask
            time_mask: optional time attention mask (unused)
            time_local_window_size: optional local window size for time attention (left, right) or int for symmetric
            deterministic: whether to use deterministic mode (no dropout)
            caches: optional dict mapping layer_index -> KVCache
            rngs: optional RNG state for dropout
            return_weights: if True, all_weights is populated with per-layer attention matrices

        Returns:
            (x, new_caches, all_weights) — all_weights is None when return_weights=False.
        """
        new_caches = {} if caches is not None else None
        all_weights: list | None = [] if return_weights else None
        x0 = x
        time_index = 0

        for i, layer in enumerate(self.layers):
            if self.use_residual_lambdas:
                resid_lambda = self.resid_lambdas.value[i].astype(self.dtype)
                x0_lambda = self.x0_lambdas.value[i].astype(self.dtype)
                x = resid_lambda * x + x0_lambda * x0

            is_time_layer = layer.is_time_layer
            cache_i = caches.get(time_index) if caches is not None and is_time_layer else None

            x, new_cache_i, w_i = layer(x, space_mask=space_mask, time_mask=time_mask, time_local_window_size=time_local_window_size, deterministic=deterministic, cache=cache_i, rngs=rngs, return_weights=return_weights)
            if all_weights is not None:
                all_weights.append(w_i)

            if new_caches is not None and new_cache_i is not None:
                new_caches[time_index] = new_cache_i
            if is_time_layer:
                time_index += 1

        return x, new_caches, all_weights

    def estimate_attention_flops(self, batch_size: int, seq_time: int, seq_space: int,
                                 time_window: int | None = None) -> int:
        """Attention FLOPs per training step (forward + backward).

        Computes FLOPs for Q@K^T and attn@V operations only (not weight matrices).
        Factor of 12 = 2 matmuls × 2 ops × 3 (forward + backward).
        """
        n_time = sum(layer.is_time_layer for layer in self.layers)
        n_space = self.depth - n_time
        t_eff = seq_time if time_window is None else min(seq_time, time_window)
        space_attn = 12 * n_space * self.d_model * (seq_space ** 2) * batch_size * seq_time
        time_attn = 12 * n_time * self.d_model * seq_time * t_eff * batch_size * seq_space
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
        self.cfg = cfg
        self.n_latents = cfg.n_latents
        self.patch_size = cfg.patch_size
        self.context_length = cfg.context_length
        self.dataset_mean = cfg.dataset_mean
        self.dataset_std = cfg.dataset_std
        self.context_length = cfg.context_length
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        self.patch_proj = nnx.Linear(cfg.patch_size * cfg.patch_size * 3, cfg.d_model, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)
        self.bottleneck_proj = nnx.Linear(cfg.d_model, cfg.d_bottleneck, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, depth=cfg.depth,
            dropout_rate=cfg.dropout_rate, qk_norm_type=cfg.qk_norm_type, mlp_ratio=4.0,
            time_every=cfg.time_every, time_layer_offset=getattr(cfg, 'time_layer_offset', 1), rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias, use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, 'use_seq_parallel', False),
            dtype=dtype, param_dtype=param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules, rngs=rngs
        )

        self.mask_and_replace = MAEReplacer(D=cfg.d_model, p_min=cfg.mae_p_min, p_max=cfg.mae_p_max, dtype=dtype, param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs)
        self.latents_enc = nnx.Param(jax.random.normal(rngs.params(), (cfg.n_latents, cfg.d_model), dtype=param_dtype) * 0.02, sharding_names=mesh_rules('embed'))

    def get_token_layout(self, H: int, W: int) -> TokenLayout:
        n_patches = (H // self.patch_size) * (W // self.patch_size)
        return TokenLayout(((Modality.LATENT, self.n_latents), (Modality.IMAGE, n_patches)))

    def create_static_caches(self, batch_size: int, H: int, W: int, window_size: int = 1024, dtype=jnp.float32) -> KVCachesDict:
        """Creates concrete, zero-filled encoder KV cache buffers for JIT compilation."""
        layout = self.get_token_layout(H, W)

        return create_transformer_caches(
            layer_is_time=[layer.is_time_layer for layer in self.transformer.layers],
            flattened_batch_size=batch_size * layout.S,
            window_size=window_size,
            num_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            dtype=dtype,
        )

    def __call__(self, videos, *, deterministic: bool = True, caches: KVCachesDict | None = None, rngs: nnx.Rngs | None = None, mae_p_max: jnp.ndarray | None = None):
        # Videos in the [0, 255] range
        B, T, H, W, C = videos.shape

        normalized_videos = normalize_with_dataset_stats(videos, mean=self.dataset_mean, std=self.dataset_std)
        patch_tokens = patchify(normalized_videos, patch=self.patch_size)
        proj_patches = self.patch_proj(patch_tokens)  # (B, T, Np, D)

        # MAE mask-and-replace on patch tokens (encoder input only during training)
        if deterministic or rngs is None:
            # Skip MAE masking during inference
            proj_patches_masked = proj_patches
            patch_mask = jnp.zeros((B, T, proj_patches.shape[2], 1), dtype=jnp.bool_)
            keep_prob = jnp.ones((B, T, 1))
        else:
            proj_patches_masked, patch_mask, keep_prob = self.mask_and_replace(proj_patches, rngs=rngs, p_max_override=mae_p_max)

        # patch_mask is (B,T,Np,1), need to expand to pixels (B, T, Np, P*P)
        patch_mask_expanded = jnp.repeat(patch_mask, self.patch_size**2, axis=-1)
        frame_mask = unpatchify(patch_mask_expanded, self.patch_size, H, W)

        # Prepend learned latents
        latents = repeat(self.latents_enc.value.astype(proj_patches.dtype), "... -> b t ...", b=B, t=T)
        tokens = jnp.concatenate([latents, proj_patches_masked], axis=2)  # (B, T, S=(Np+Nl), D)

        layout = self.get_token_layout(H, W)
        space_mask = layout.build_space_mask("encoder")  # (1, 1, q_len, k_len)

        # Feed tokens into transformer
        time_local_window_size = (self.context_length - 1, 0) if self.context_length is not None else None
        encoded_tokens, new_caches, _ = self.transformer(tokens, space_mask=space_mask, time_local_window_size=time_local_window_size, deterministic=deterministic, caches=caches, rngs=rngs)

        # Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, :self.n_latents]
        proj_tokens = nnx.tanh(self.bottleneck_proj(latent_tokens))

        # new_caches is None unless a KV cache was provided.
        return proj_tokens, (frame_mask, keep_prob), new_caches


class Decoder(nnx.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.
    """

    def __init__(self, cfg: DecoderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg
        self.n_latents = cfg.n_latents
        self.patch_size = cfg.patch_size
        self.context_length = cfg.context_length
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.H = cfg.H
        self.W = cfg.W
        self.dataset_mean = cfg.dataset_mean
        self.dataset_std = cfg.dataset_std
        self.context_length = cfg.context_length
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        self.n_patches = (self.H // self.patch_size) * (self.W // self.patch_size)
        self.up_proj = nnx.Linear(cfg.d_bottleneck, cfg.d_model, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')), rngs=rngs)
        self.patch_head = nnx.Linear(cfg.d_model, cfg.d_patch, use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype, kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')), rngs=rngs)

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, depth=cfg.depth,
            dropout_rate=cfg.dropout_rate, qk_norm_type=cfg.qk_norm_type, mlp_ratio=4.0,
            time_every=cfg.time_every, time_layer_offset=getattr(cfg, 'time_layer_offset', 1), rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias, use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, 'use_seq_parallel', False),
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

    def create_static_caches(self, batch_size: int, window_size: int = 1024, dtype=jnp.float32) -> KVCachesDict:
        """Creates concrete, zero-filled decoder KV cache buffers for JIT compilation."""
        layout = self.get_token_layout()

        return create_transformer_caches(
            layer_is_time=[layer.is_time_layer for layer in self.transformer.layers],
            flattened_batch_size=batch_size * layout.S,
            window_size=window_size,
            num_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            dtype=dtype,
        )

    def __call__(
            self,
            z: jnp.ndarray,
            *,
            deterministic: bool = True,
            caches: KVCachesDict | None = None,
            rngs: nnx.Rngs | None = None
        ):
        # Always expect unpacked input: (B, T, n_latents, d_bottleneck)
        B, T, N_l, d_bottleneck = z.shape

        # Up-project latent bottleneck to d_model (per latent token)
        latents = self.up_proj(z)  # (B, T, N_l, D)

        # Learned per-patch query tokens (owned by the decoder)
        patches = repeat(self.patch_queries.value.astype(latents.dtype), " ... -> b t ...", b=B, t=T)  # (B, T, Np, D)

        # Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=-2)

        # Make mask
        layout = self.get_token_layout()
        space_mask = layout.build_space_mask("decoder")

        time_local_window_size = (self.context_length - 1, 0) if self.context_length is not None else None
        x, new_caches, _ = self.transformer(tokens, space_mask=space_mask, time_local_window_size=time_local_window_size, deterministic=deterministic, caches=caches, rngs=rngs)

        # Prediction head over the patch-query slice
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

    def __call__(self, videos, *, deterministic: bool = True, caches: TokenizerKVCachesDict | None = None, rngs: nnx.Rngs | None = None, mae_p_max: jnp.ndarray | None = None):
        encoder_caches = caches.get("encoder") if caches is not None else None
        decoder_caches = caches.get("decoder") if caches is not None else None
        z, aux, encoder_caches = self.encoder(videos, deterministic=deterministic, caches=encoder_caches, rngs=rngs, mae_p_max=mae_p_max)
        recon, decoder_caches = self.decoder(z, deterministic=deterministic, caches=decoder_caches, rngs=rngs)
        return recon, aux, {"encoder": encoder_caches, "decoder": decoder_caches}

    def encode(self, videos, *, deterministic: bool = True, caches: KVCachesDict | None = None, rngs: nnx.Rngs | None = None, mae_p_max: jnp.ndarray | None = None):
        # Always returns unpacked: (B, T, n_latents, d_bottleneck)
        return self.encoder(videos, deterministic=deterministic, caches=caches, rngs=rngs, mae_p_max=mae_p_max)

    def decode(self, z, *, deterministic: bool = True, caches: KVCachesDict | None = None, rngs: nnx.Rngs | None = None):
        # Always expects unpacked: (B, T, n_latents, d_bottleneck)
        return self.decoder(z, deterministic=deterministic, caches=caches, rngs=rngs)

    def create_encoder_static_caches(self, batch_size: int, H: int, W: int, window_size: int = 1024, dtype=jnp.float32) -> KVCachesDict:
        """Creates concrete, zero-filled encoder KV cache buffers for JIT compilation."""
        return self.encoder.create_static_caches(batch_size=batch_size, H=H, W=W, window_size=window_size, dtype=dtype)

    def create_decoder_static_caches(self, batch_size: int, window_size: int = 1024, dtype=jnp.float32) -> KVCachesDict:
        """Creates concrete, zero-filled decoder KV cache buffers for JIT compilation."""
        return self.decoder.create_static_caches(batch_size=batch_size, window_size=window_size, dtype=dtype)

    def create_static_caches(
            self,
            batch_size: int,
            window_size: int = 1024,
            dtype=jnp.float32,
        ) -> KVCachesDict:
        """Creates decoder KV caches. Kept for backward compatibility."""
        return self.create_decoder_static_caches(batch_size=batch_size, window_size=window_size, dtype=dtype)

    def create_tokenizer_static_caches(
            self,
            batch_size: int,
            H: int,
            W: int,
            window_size: int = 1024,
            dtype=jnp.float32,
        ) -> TokenizerKVCachesDict:
        """Creates namespaced encoder and decoder KV cache buffers."""
        encoder_window_size = self.encoder.context_length if self.encoder.context_length is not None else window_size

        return {
            "encoder": self.create_encoder_static_caches(batch_size=batch_size, H=H, W=W, window_size=encoder_window_size, dtype=dtype),
            "decoder": self.create_decoder_static_caches(batch_size=batch_size, window_size=window_size, dtype=dtype),
        }

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

        Follows https://github.com/karpathy/nanochat/discussions/420.
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

    def __init__(
        self,
        d_model: int,
        num_binary_actions: int = 0,
        categorical_action_dim: int = 0,
        continuous_action_dim: int = 0,
        use_bias: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs
    ):
        self.d_model = d_model
        dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)
        self.dtype = dtype

        # Base "action token" embedding (used always)
        self.base_action_emb = nnx.Param(
            jax.random.normal(rngs.params(), (d_model,), dtype=param_dtype) * 0.02,
            sharding_names=mesh_rules('embed')
        )

        # Embed binary actions
        if num_binary_actions > 0:
            binary_embeds = [
                nnx.Embed(
                    2, d_model,
                    dtype=dtype, param_dtype=param_dtype,
                    embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=1.0), mesh_rules('embed')),
                    rngs=rngs
                )
                for _ in range(num_binary_actions)
            ]
            self.binary_embeds_list = _nnx_list(binary_embeds)  # num_binary_actions * (B, T, d_model)
        else:
            self.binary_embeds_list = None

        # Embed categorical action
        if categorical_action_dim > 0:
            self.categorical_embeds = nnx.Embed(
                categorical_action_dim, d_model,
                dtype=dtype, param_dtype=param_dtype,
                embedding_init=nnx.with_partitioning(nnx.initializers.normal(stddev=1.0), mesh_rules('embed')),
                rngs=rngs
            )  # (B, T, d_model)
        else:
            self.categorical_embeds = None

        # Continuous actions: linear projection
        if continuous_action_dim > 0:
            self.continuous_proj = nnx.Linear(
                continuous_action_dim, d_model,
                use_bias=use_bias,
                dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')),
                rngs=rngs
            )  # (B, T, d_model)
        else:
            self.continuous_proj = None

    def __call__(
        self,
        actions: Actions,
        batch_time_shape: Tuple[int, int],
        as_tokens: bool = True
    ):
        """
        Encode actions into embeddings.

        Args:
            actions: Actions to encode (B, T, ...)
            batch_time_shape: (B, T) shape for unlabeled videos
            as_tokens: whether to expand to token dimension

        Returns:
            Action embeddings (B, T, d_model) or (B, T, 1, d_model) if as_tokens=True
        """
        base_emb = self.base_action_emb.value.astype(self.dtype)

        # Unlabeled videos: just broadcast base embedding
        B, T = batch_time_shape
        out = jnp.broadcast_to(base_emb, (B, T, self.d_model))

        if actions.binary is not None and self.binary_embeds_list is not None:
            binary_emb = jnp.zeros((B, T, self.d_model), dtype=self.dtype)
            for i, emb in enumerate(self.binary_embeds_list):
                binary_state = actions.binary[..., i]  # (B, T)
                binary_emb = binary_emb + emb(binary_state)
            out = out + binary_emb

        if actions.categorical is not None and self.categorical_embeds is not None:
            categorical_emb = self.categorical_embeds(actions.categorical)
            out = out + categorical_emb

        if actions.continuous is not None and self.continuous_proj is not None:
            continuous_emb = self.continuous_proj(actions.continuous.astype(self.dtype))
            out = out + continuous_emb

        if as_tokens:
            # expand a token axis (S_a = 1)
            out = out[..., None, :]

        return out


class TimestepEmbedder(nnx.Module):
    """Sinusoidal timestep embedding"""

    def __init__(
        self,
        out_dim: int,
        freq_dim: int = 256,
        max_period: float = 10000.0,
        *,
        dtype,
        param_dtype,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.dtype = dtype
        self.dense1 = nnx.Linear(
            freq_dim, out_dim,
            use_bias=True,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(0.02), mesh_rules('mlp')),
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            out_dim, out_dim,
            use_bias=True,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(0.02), mesh_rules('mlp')),
            dtype=dtype, param_dtype=param_dtype, rngs=rngs,
        )

    def __call__(self, t: jax.Array) -> jax.Array:
        """Embed scalar timestep(s).

        Args:
            t: (...) float array of timestep values.

        Returns:
            (..., out_dim) embedding array.
        """
        t = t.astype(jnp.float32) * 1000.0  # scale so t spans a useful range of the sinusoidal basis (DiT convention)
        half = self.freq_dim // 2
        freqs = jnp.exp(-math.log(self.max_period) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[..., None] * freqs  # (..., half)
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)  # (..., freq_dim)
        emb = emb.astype(self.dtype)
        emb = nnx.silu(self.dense1(emb))
        return self.dense2(emb)


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
        self.action_encoder = ActionEncoder(
            d_model=cfg.d_model,
            num_binary_actions=cfg.num_binary_actions,
            categorical_action_dim=cfg.categorical_action_dim,
            continuous_action_dim=cfg.continuous_action_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs
        )

        # Transformer
        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            depth=cfg.depth, dropout_rate=cfg.dropout_rate, qk_norm_type=cfg.qk_norm_type,
            mlp_ratio=cfg.mlp_ratio, time_every=cfg.time_every, time_layer_offset=cfg.time_layer_offset, rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias, use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, 'use_seq_parallel', False),
            dtype=self.dtype, param_dtype=self.param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules, rngs=rngs
        )

        # Sinusoidal embeddings for shortcut conditioning
        # step_idx ∈ {0,...,log2(k_max)} passed as float; tau ∈ [0,1] (tau_idx/k_max)
        half_dim = cfg.d_model // 2
        self.step_embed = TimestepEmbedder(
            out_dim=half_dim,
            dtype=self.dtype, param_dtype=self.param_dtype,
            mesh_rules=mesh_rules, rngs=rngs,
        )
        self.signal_embed = TimestepEmbedder(
            out_dim=half_dim,
            dtype=self.dtype, param_dtype=self.param_dtype,
            mesh_rules=mesh_rules, rngs=rngs,
        )

        # Output head (zero-init)
        self.flow_x_head = nnx.Linear(
            cfg.d_model, cfg.d_bottleneck * cfg.packing_factor,
            use_bias=cfg.use_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
            bias_init=nnx.initializers.zeros,
            dtype=self.dtype, param_dtype=self.param_dtype, rngs=rngs
        )

    def get_token_layout(self, n_latents: int, n_agent: int = 0) -> TokenLayout:
        """Get token layout.

        Args:
            n_latents: Number of unpacked latent tokens (will be packed internally)
            n_agent: Number of agent tokens
        """
        n_spatial = n_latents // self.packing_factor
        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT, 1),
            (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, self.cfg.n_register),
        ]
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))
        return TokenLayout(tuple(segments))

    def create_static_caches(self, batch_size: int, n_latents: int, window_size: int = 1024,
                           n_agent: int = 0, dtype=jnp.float32) -> KVCachesDict:
        """Creates concrete, zero-filled buffers for JIT compilation.

        Args:
            batch_size: Batch size
            n_latents: Number of unpacked latent tokens (will be packed internally)
            window_size: Maximum temporal sequence length
            n_agent: Number of agent tokens
            dtype: Data type for cache buffers
        """
        layout = self.get_token_layout(n_latents=n_latents, n_agent=n_agent)

        return create_transformer_caches(
            layer_is_time=[layer.is_time_layer for layer in self.transformer.layers],
            flattened_batch_size=batch_size * layout.S,
            window_size=window_size,
            num_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            dtype=dtype
        )

    def __call__(
        self,
        actions: Actions,
        step_indices,
        tau_indices,
        unpacked_enc_tokens,
        *,
        context_length: int | None = None,
        time_mask: jnp.ndarray | None = None,
        task_embeddings: jnp.ndarray | None = None,
        deterministic: bool = True,
        caches: KVCachesDict | None = None,
        rngs: nnx.Rngs | None = None
    ) -> Tuple[jax.Array, Tuple[jax.Array | None, KVCachesDict | None]]:
        """
        Args:
            actions: Actions object
            step_indices: (B, T) int32 — step indices for embedding lookup
            tau_indices: (B, T) int32 - signal indices for embedding lookup
            unpacked_enc_tokens: (B, T, n_latents, d_bottleneck) unpacked encoder tokens
            context_length: optional context length for sliding window attention. If provided,
                           creates local_window_size=(context_length - 1, 0) for causal sliding window.
            time_mask: optional (B, 1, T, T) boolean mask for temporal attention
            task_embeddings: (B, T, n_agent, d_model) optional agent tokens
            caches: optional dict of KVCache for each layer

        Shapes produced (internal):
            spatial_tokens: (B, T, n_spatial, d_model)
            action_token:  (B, T, 1, d_model)
            shortcut_token: (B, T, 1, d_model)
        """
        B, T = unpacked_enc_tokens.shape[:2]

        # Pack tokens internally for processing
        packed_enc_tokens = rearrange(unpacked_enc_tokens, "b t (n p) d -> b t n (p d)", p=self.packing_factor)
        # Project spatial tokens to d_model
        spatial_tokens = self.spatial_proj(packed_enc_tokens)  # (B, T, n_spatial, d_model)

        # Encode actions to d_model
        action_token = self.action_encoder(
            actions=actions,
            batch_time_shape=(B, T),
        )  # (B, T, 1, d_model)

        # Prepare learned register tokens
        B, T = spatial_tokens.shape[:2]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value.astype(self.dtype)[None, None, ...],  # (1, 1 ,n_register, d_model)
            (B, T, self.n_register, self.d_model),
        )

        # Shortcut embeddings (sinusoidal, concatenated to single token)
        # step_indices: int log2(K) → float for sinusoidal PE
        # tau_indices: int τ*k_max → normalize to [0,1] for sinusoidal PE
        step_emb = self.step_embed(step_indices.astype(jnp.float32))                          # (B, T, d_model//2)
        signal_emb = self.signal_embed(tau_indices.astype(jnp.float32) / self.k_max)          # (B, T, d_model//2)
        shortcut_token = jnp.concatenate([step_emb, signal_emb], axis=-1)[:, :, None, :]     # (B, T, 1, d_model)

        # Concatenate in declared layout order (`get_token_layout`)
        tokens = [action_token, shortcut_token, spatial_tokens, register_tokens]
        if task_embeddings is not None:
            tokens.append(task_embeddings)

        tokens = jnp.concatenate(tokens, axis=2)  # (B, T, S, D)

        # Make the layout for masking
        n_agent = task_embeddings.shape[2] if task_embeddings is not None else 0
        n_latents = unpacked_enc_tokens.shape[2]
        layout = self.get_token_layout(n_latents=n_latents, n_agent=n_agent)
        space_mask = layout.build_space_mask("wm_agent")

        # Compute local_window_size from context_length for sliding window causal attention
        time_local_window_size = None
        if context_length is not None:
            time_local_window_size = (context_length - 1, 0)

        x, new_caches, _ = self.transformer(
            tokens, space_mask=space_mask,
            time_mask=time_mask,
            time_local_window_size=time_local_window_size,
            deterministic=deterministic,
            caches=caches,
            rngs=rngs
        )

        spatial_tokens = x[:, :, layout.slices()[Modality.SPATIAL], :]
        x1_hat_packed = self.flow_x_head(spatial_tokens)  # (B, T, n_spatial, d_spatial)

        # Unpack before returning
        x1_hat = rearrange(x1_hat_packed, "b t n (p d) -> b t (n p) d", p=self.packing_factor)
        
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
        if self.action_encoder.binary_embeds_list is not None:
            for binary_emb in self.action_encoder.binary_embeds_list:
                excluded += binary_emb.embedding.value.size
        if self.action_encoder.categorical_embeds is not None:
            excluded += self.action_encoder.categorical_embeds.embedding.value.size
        # Per-layer scalars
        excluded += self.transformer.count_excluded_params()
        return excluded

    def estimate_flops(self, batch_size: int, seq_length: int, n_latents: int) -> int:
        """FLOPs per training step (forward + backward).

        Args:
            batch_size: Batch size
            seq_length: Sequence length
            n_latents: Number of unpacked latent tokens (will be packed internally)

        Follows https://github.com/karpathy/nanochat/discussions/420.
        """
        S = self.get_token_layout(n_latents=n_latents).S

        # Weight FLOPs (excluding embeddings/scalars)
        total_params = self.num_scaling_params()
        excluded = self.count_excluded_params()
        weight_flops = 6 * (total_params - excluded) * batch_size * seq_length * S

        # Attention FLOPs
        attn_flops = self.transformer.estimate_attention_flops(batch_size, seq_length, S, self.cfg.context_length)

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
        self.L = cfg.L
        self.num_binary_actions = cfg.num_binary_actions
        self.categorical_action_dim = cfg.categorical_action_dim
        self.continuous_action_dim = cfg.continuous_action_dim
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        # Feature projector: operates on flattened agent tokens (L*d_model)
        d_flat = cfg.d_model * cfg.L
        self.projector = MLP(
            d_model=d_flat, mlp_ratio=cfg.mlp_ratio, dropout_rate=cfg.dropout_rate,
            swiglu=cfg.swiglu, parity_2over3=cfg.parity_2over3, use_bias=cfg.use_bias, dtype=dtype,
            param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs
        )

        self.out_binary = None
        if cfg.num_binary_actions > 0:
            # Binary keyboard: one logit per key (Bernoulli)
            self.out_binary = nnx.Linear(
                d_flat, cfg.L * cfg.num_binary_actions,
                use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
                rngs=rngs
            )

        self.out_categorical = None
        if cfg.categorical_action_dim > 0:
            self.out_categorical = nnx.Linear(
                d_flat, cfg.L * cfg.categorical_action_dim,
                use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
                rngs=rngs
            )

        self.out_continuous = None
        if cfg.continuous_action_dim > 0:
            # Gaussian: mean + log_var
            self.out_continuous = nnx.Linear(
                d_flat, cfg.L * cfg.continuous_action_dim * 2,
                use_bias=cfg.use_bias, dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
                rngs=rngs
            )

    def __call__(
        self,
        h_t: jnp.ndarray,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None
    ) -> dict[str, jnp.ndarray]:
        """
        Forward pass.

        Args:
            h_t: (B, T, n_agent, d_model) hidden states

        Returns:
            Dict with keys:
            - "binary_logits": (B, T, L, num_binary_actions) if binary actions enabled
            - "categorical_logits": (B, T, L) if categorical action enabled
            - "mouse_mean": (B, T, L, continuous_action_dim) if continuous action enabled
            - "mouse_log_var": (B, T, L, continuous_action_dim) if continuous action enabled
        """
        h_t = einops.rearrange(h_t, 'b t n c -> b t (n c)')
        x = self.projector(h_t, deterministic=deterministic, rngs=rngs)  # (B, T, D)

        outputs = {}

        if self.out_binary is not None:
            binary_logits = self.out_binary(x)
            binary_logits = rearrange(binary_logits, "b t (l a) -> b t l a", l=self.cfg.L, a=self.num_binary_actions)
            outputs["binary_logits"] = binary_logits

        if self.out_categorical is not None:
            categorical_logits = self.out_categorical(x)
            categorical_logits = rearrange(categorical_logits, "b t (l a) -> b t l a", l=self.cfg.L, a=self.categorical_action_dim)
            outputs["categorical_logits"] = categorical_logits

        if self.out_continuous is not None:
            continuous_out = self.out_continuous(x)
            continuous_out = rearrange(continuous_out, "b t (l m) -> b t l m", l=self.cfg.L, m=self.cfg.continuous_action_dim * 2)
            outputs["continuous_mean"] = continuous_out[..., :self.cfg.continuous_action_dim]
            outputs["continuous_log_var"] = continuous_out[..., self.cfg.continuous_action_dim:]

        return outputs

    def sample(
        self,
        h_t: jnp.ndarray,
        *,
        deterministic: bool = True,
        rng: jax.Array,
    ) -> Actions:
        """
        Sample actions from the policy.

        Args:
            h_t: (B, T, n_agent, d_model) hidden states
            deterministic: whether to do deterministic sampling (greedy)
            rng: Random key for sampling

        Returns: Action object with actions sampled from policy
        """
        rng_policy, rng_binary, rng_categorical, rng_continuous = jax.random.split(rng, num=4)

        outputs = self(h_t, deterministic=deterministic, rngs=rng_policy)
        actions = Actions()

        if "binary_logits" in outputs:
            binary_logits = outputs["binary_logits"]
            if deterministic:
                actions.binary = (binary_logits > 0).astype(jnp.int32)
            else:
                probs = jax.nn.sigmoid(binary_logits)
                actions.binary = jax.random.bernoulli(rng_binary, probs).astype(jnp.int32)

        if "categorical_logits" in outputs:
            categorical_logits = outputs["categorical_logits"]
            if deterministic:
                actions.categorical = jnp.argmax(categorical_logits, axis=-1)
            else:
                actions.categorical = jax.random.categorical(rng_categorical, categorical_logits, axis=-1)

        if "continuous_mean" in outputs:
            mean = outputs["continuous_mean"]
            if deterministic:
                actions.continuous = mean
            else:
                log_var = outputs["continuous_log_var"]
                std = jnp.exp(0.5 * log_var)
                eps = jax.random.normal(rng_continuous, mean.shape)
                actions.continuous = mean + std * eps

        return actions

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

    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True, rngs: nnx.Rngs | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
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


    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True, rngs: nnx.Rngs | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        h_t = einops.rearrange(h_t, 'b t n c -> b t (n c)')
        x = self.projector(h_t, deterministic=deterministic, rngs=rngs)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, K)
        return logits, self.symexp_centers_log
