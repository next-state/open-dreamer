import jax.numpy as jnp
from flax import nnx
import jax
from typing import Tuple, Any, Dict, NamedTuple, Sequence
from einops import rearrange, repeat
import math
from .utils import (
    Modality,
    TokenLayout,
    normalize_with_dataset_stats,
    unnormalize_with_dataset_stats,
    to_jnp_dtype,
    patchify,
    unpatchify,
)
from .configs import (
    TokenizerModelConfig,
    DynamicsModelConfig,
    EncoderModelConfig,
    DecoderModelConfig,
    TextEncoderConfig,
    TextEncoderModelConfig,
)
from .parallel import MeshRules
from .actions import Actions


# compatibility for reactor runtime. The reactor runtime has to use the older Flax version that does not have nnx.List
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
        self.k = k  # (B, Max_T, K, H)
        self.v = v  # (B, Max_T, K, H)
        self.index = index  # scalar integer (i32)
        self.window_size = window_size  # static int

    def tree_flatten(self):
        children = (self.k, self.v, self.index)
        aux_data = (self.window_size,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        k, v, index = children
        (window_size,) = aux_data
        return cls(k, v, index, window_size)

    @classmethod
    def init(cls, batch_size, window_size, num_kv_heads, head_dim, dtype=jnp.float32):
        dtype = to_jnp_dtype(dtype)
        return cls(
            k=jnp.zeros((batch_size, window_size, num_kv_heads, head_dim), dtype=dtype),
            v=jnp.zeros((batch_size, window_size, num_kv_heads, head_dim), dtype=dtype),
            index=jnp.array(0, dtype=jnp.int32),
            window_size=window_size,
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
            fits_contiguous, _update_contiguous, _update_wrap, self.k, k_new, write_idx
        )
        v_updated = jax.lax.cond(
            fits_contiguous, _update_contiguous, _update_wrap, self.v, v_new, write_idx
        )

        return KVCache(
            k=k_updated, v=v_updated, index=self.index + T, window_size=self.window_size
        )

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
        k_idx = jnp.arange(self.window_size)[None, None, None, :]  # (1, 1, 1, Window)
        start_valid = self.window_size - valid_len
        valid_mask = k_idx >= start_valid

        # Shifted causal mask
        q_idx = jnp.arange(query_len)[None, None, :, None]
        causal_mask = k_idx <= (self.window_size - query_len + q_idx)

        final_mask = jnp.logical_and(valid_mask, causal_mask)

        return k_ordered, v_ordered, final_mask


KVCachesDict = Dict[int, KVCache]  # Type alias for transformer KV cache dictionaries


class TokenizerCaches(NamedTuple):
    """Namespaced KV caches for the tokenizer encoder and decoder."""

    encoder: KVCachesDict | None
    decoder: KVCachesDict | None


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
                dtype=dtype,
            )
            time_index += 1
    return caches


# ============================================================================
# Building Blocks
# ============================================================================


class RotaryEmbedding1D(nnx.Module):
    """Rotary Position Embedding with precomputed frequencies."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
    ):
        self.dim = dim
        self.theta = theta
        self.dtype = to_jnp_dtype(dtype)
        self.param_dtype = to_jnp_dtype(param_dtype)

        # Precompute inverse frequencies as a Variable (not a Param, so it won't be trained)
        inv_freq = 1.0 / (
            self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        self.inv_freq = nnx.Variable(inv_freq)

    def __call__(self, q, k, start_pos=0):
        """
        q: (B, T, N, H)
        k: (B, T, K, H)
        start_pos: scalar int (the absolute position of the first frame in T)
        """
        T = q.shape[1]
        t_indices = jnp.arange(T, dtype=self.dtype) + start_pos
        freqs = jnp.outer(t_indices, self.inv_freq.value)  # (T, dim//2)

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
        xq_pairs = rearrange(xq, "... (d two) -> ... d two", two=2)
        xk_pairs = rearrange(xk, "... (d two) -> ... d two", two=2)

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

        xq_out = rearrange(xq_out, "... d two -> ... (d two)")
        xk_out = rearrange(xk_out, "... d two -> ... (d two)")

        return xq_out, xk_out


class MAEReplacer(nnx.Module):
    """Masked Autoencoder token replacer for training."""

    def __init__(
        self,
        D: int,
        p_min: float = 0.0,
        p_max: float = 0.9,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
        self.p_min = p_min
        self.p_max = p_max
        self.dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        # Learnable mask token
        self.mask_token = nnx.Param(
            jax.random.normal(rngs.params(), (D,), dtype=param_dtype) * 0.02,
            sharding_names=mesh_rules("embed"),
        )

    def __call__(
        self,
        patches_btnd: jnp.ndarray,
        *,
        rngs: nnx.Rngs,
        p_max_override: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # patches_btnd: (B,T,Np,D)
        B, T, Np, D = patches_btnd.shape
        mask_token = self.mask_token.value.astype(self.dtype)

        p_max = p_max_override if p_max_override is not None else self.p_max

        # Draw RNGs
        p_rng = rngs.mae()
        m_rng = rngs.mae()
        p_bt = jax.random.uniform(
            p_rng, (B, T), minval=self.p_min, maxval=p_max
        )  # (B,T)
        keep_prob_bt1 = 1.0 - p_bt[..., None]  # (B,T,1)
        keep = jax.random.bernoulli(m_rng, keep_prob_bt1, (B, T, Np))  # (B,T,Np)
        keep = keep[..., None]  # (B,T,Np,1)
        replaced = jnp.where(keep, patches_btnd, mask_token.reshape(1, 1, 1, D))
        mae_mask = (~keep).astype(jnp.bool_)  # (B,T,Np,1)
        return replaced, mae_mask, keep_prob_bt1


class MLP(nnx.Module):
    """Transformer MLP with optional SwiGLU gating."""

    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        swiglu: bool = True,
        parity_2over3: bool = False,
        use_norm: bool = True,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
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
            self.norm = nnx.RMSNorm(
                d_model,
                use_scale=self.use_rmsnorm_scale,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        if self.swiglu:
            self.fc_in = nnx.Linear(
                d_model,
                2 * hidden,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), mesh_rules("mlp")
                ),
                rngs=rngs,
            )
        else:
            self.fc_in = nnx.Linear(
                d_model,
                hidden,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), mesh_rules("mlp")
                ),
                rngs=rngs,
            )

        self.fc_out = nnx.Linear(
            hidden,
            d_model,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate)

    def __call__(
        self, x: jnp.ndarray, deterministic: bool = True, rngs: nnx.Rngs | None = None
    ) -> jnp.ndarray:
        if self.use_norm:
            x = self.norm(x)

        if self.swiglu:
            # SwiGLU: Dense -> split -> u * silu(v)
            pre = self.fc_in(x)  # (..., 2H)
            u, v = jnp.split(pre, 2, axis=-1)  # (..., H), (..., H)
            h = u * jax.nn.silu(v)  # (..., H)
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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        is_causal: bool = False,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
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

        if use_seq_parallel:
            raise NotImplementedError("Sequence parallel not implemented for GQA")

        assert self.dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        self.to_q = nnx.Linear(
            dim,
            dim,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("attn")
            ),
            rngs=rngs,
        )
        self.to_kv = nnx.Linear(
            dim,
            2 * kv_dim,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("attn")
            ),
            rngs=rngs,
        )
        self.to_out = nnx.Linear(
            dim,
            dim,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("attn")
            ),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate)

        if self.qk_norm_type == "qknorm":
            self.q_ln = nnx.RMSNorm(
                head_dim,
                use_scale=self.use_rmsnorm_scale,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.k_ln = nnx.RMSNorm(
                head_dim,
                use_scale=self.use_rmsnorm_scale,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.rope = RotaryEmbedding1D(
            dim=head_dim, theta=self.rope_theta, dtype=dtype, param_dtype=param_dtype
        )

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
        if self.qk_norm_type == "qknorm":
            q = self.q_ln(q).astype(self.dtype)
            k = self.k_ln(k).astype(self.dtype)
        elif self.qk_norm_type == "quest":
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
            q,
            k_attn,
            v_attn,
            mask=mask_attn,
            scale=scale,
            is_causal=attn_is_causal,
            local_window_size=local_window_size if cache is None else None,
        )

        if return_weights:
            G = self.num_heads // self.num_kv_heads
            k_exp = jnp.repeat(k_attn, G, axis=2)  # (B, S, N, H)
            logits = jnp.einsum("B T N H, B S N H -> B N T S", q, k_exp) * scale
            if mask_attn is not None:
                logits = jnp.where(mask_attn, logits, jnp.finfo(logits.dtype).min)
            elif attn_is_causal:
                T_q, T_k = q.shape[1], k_attn.shape[1]
                causal = jnp.tril(jnp.ones((T_q, T_k), dtype=jnp.bool_))
                logits = jnp.where(
                    causal[None, None], logits, jnp.finfo(logits.dtype).min
                )
            attn_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        else:
            attn_weights = None

        attn = rearrange(attn, "B T N H -> B T (N H)")

        out = self.to_out(attn)
        out = self.dropout(out, deterministic=deterministic, rngs=rngs)

        return out, new_cache, attn_weights


class GroupedCrossAttention(nnx.Module):
    """Grouped Query cross-attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        is_causal: bool = False,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
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

        if use_seq_parallel:
            raise NotImplementedError("Sequence parallel not implemented for GQA")

        assert self.dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        self.kv_dim = kv_dim
        self.num_groups = self.num_heads // self.num_kv_heads

        self.to_q = nnx.Linear(
            dim,
            dim,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("attn")
            ),
            rngs=rngs,
        )
        self.to_kv = nnx.Linear(
            dim,
            2 * kv_dim,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("attn")
            ),
            rngs=rngs,
        )
        self.to_out = nnx.Linear(
            dim,
            dim,
            use_bias=self.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("attn")
            ),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate)

        if self.qk_norm_type == "qknorm":
            self.q_ln = nnx.RMSNorm(
                head_dim,
                use_scale=self.use_rmsnorm_scale,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.k_ln = nnx.RMSNorm(
                head_dim,
                use_scale=self.use_rmsnorm_scale,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.rope = RotaryEmbedding1D(
            dim=head_dim, theta=self.rope_theta, dtype=dtype, param_dtype=param_dtype
        )

    def __call__(
        self,
        x,
        X,
        text_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
        return_weights: bool = False,
    ):
        """
        x = patch tokens (queries), already flattened: (B', S, D), B' = B*T
        X = text tokens (keys/values), already repeated: (B', L, D)
        text_mask = padding mask, (B', L) bool, True = real token, False = pad
        B' = batch (B*T after flattening frames into batch)
        S = number of patch tokens (query length)
        L = number of text tokens (key/value length)
        N = number of attention heads
        H = dimensions of each attention head
        K = number of key/value heads
        G = number of groups, which equals to N // K

        REVIEW ITEM 1: I originally took Q from the text and K/V from the patches.
            Whichever side supplies Q decides the output length, so I was getting
            one vector per *word* (B', L, D), which I can't add back to the video
            stream (the residual needs (B', S, D)). That direction means "the text
            reads the image". What I want for conditioning is "each patch reads the
            text", so I now take Q from the patches and K/V from the text.

            S and L do NOT have to match. The score matrix is (S, L): every query
            is compared with every key. Only the per-head width H has to agree.
        """
        G = self.num_groups

        # Q from patches, K/V from text (REVIEW ITEM 1)
        q = self.to_q(x)  # (B', S, N*H)
        kv = self.to_kv(X)  # (B', L, 2*K*H)

        # REVIEW ITEM 2: my old reshape "B T (G H_kv) -> B G T H_kv" assumed
        #     G*H == dim, but G*H = (N/K)*H = dim/K (4*64 = 256 != 512 with my
        #     defaults), so einops raised. The axis order (B, G, T, H) was also not
        #     what jax.nn.dot_product_attention expects, which is (B, T, N, H).
        #     I now split Q into all N heads. JAX does the GQA grouping for me: when
        #     q has N heads and k/v have K heads, query head n reads kv head n // G.
        q = rearrange(q, "B S (N H) -> B S N H", N=self.num_heads)
        k, v = rearrange(kv, "B L (C K H) -> C B L K H", C=2, K=self.num_kv_heads)

        scale = q.shape[-1] ** -0.5
        if self.qk_norm_type == "qknorm":
            q = self.q_ln(q).astype(self.dtype)
            k = self.k_ln(k).astype(self.dtype)
        elif self.qk_norm_type == "quest":
            # claims to beat qknorm https://openreview.net/pdf?id=HkztQWZfl2
            k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
            scale = 1.0

        # REVIEW ITEM 3: I had copied the KV-cache / is_causal / local_window_size
        #     branch from self-attention. None of it applies here: there's no causal
        #     order between patches and words, and cache.update() would have kept
        #     appending the same text keys every step. I removed it. For fast
        #     inference later, I only need to compute k, v from the text once and
        #     reuse them, since the instruction doesn't change during a rollout.

        # REVIEW ITEM 4: I was passing space_mask, which is a patch x patch (S, S)
        #     mask. Cross-attention scores are (S, L), so that was the wrong shape
        #     and the wrong meaning. What I actually need is a key-padding mask from
        #     the tokenizer's attention_mask (1 = real word, 0 = pad). It is NOT a
        #     causal mask: every patch should see every real word.
        #     (B', L) -> (B', 1, 1, L) broadcasts to (B', N heads, S queries, L keys).
        #     dot_product_attention takes a *boolean* mask (True = attend) and puts
        #     the -inf in for me.
        if text_mask is not None:
            mask_attn = text_mask.astype(jnp.bool_)[:, None, None, :]
        else:
            mask_attn = None

        # REVIEW ITEM 5: I'm deliberately not applying RoPE. Patches and words don't
        #     share a position axis, so "relative distance" between them is
        #     meaningless. Word order is already baked into the text embeddings.

        # SDPA
        attn = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask_attn,
            scale=scale,
        )  # (B', S, N, H)

        if return_weights:
            k_exp = jnp.repeat(k, G, axis=2)  # (B', L, N, H), head n -> kv head n//G
            logits = jnp.einsum("B S N H, B L N H -> B N S L", q, k_exp) * scale
            if mask_attn is not None:
                logits = jnp.where(mask_attn, logits, jnp.finfo(logits.dtype).min)
            attn_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        else:
            attn_weights = None

        attn = rearrange(attn, "B S N H -> B S (N H)")

        out = self.to_out(attn)
        out = self.dropout(out, deterministic=deterministic, rngs=rngs)

        return out, None, attn_weights


class SpaceTextCrossAttention(nnx.Module):
    """Cross-attention: patch tokens (queries) attend to text tokens (keys/values)."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        dropout_rate: float = 0.0,
        rope_theta: float = 10000.0,
        use_rmsnorm_scale: bool = True,
        use_bias: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        qk_norm_type: str | None = None,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
        assert num_heads % num_kv_heads == 0
        assert dim % num_heads == 0

        # REVIEW ITEM 6: this wrapper used to build its own to_q / to_kv / to_out /
        #     q_ln / k_ln. They were never called (self.attn has its own), but they
        #     still cost memory, optimizer state and checkpoint space, and
        #     dim // num_kv_heads isn't even a real head size. I removed them;
        #     GroupedCrossAttention owns all the projections.
        self.attn = GroupedCrossAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
            qk_norm_type=qk_norm_type,
            is_causal=False,
            use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

    def __call__(
        self,
        x,
        X,
        *,
        text_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        rngs: nnx.Rngs | None = None,
        return_weights: bool = False,
    ):
        """
        x = video patch tokens, (B, T, S, D)
        X = text tokens from the text encoder, (B, L, D)
        text_mask = tokenizer attention_mask (padding mask), (B, L), 1 = real, 0 = pad
        B = batch size, T = frames, S = patches per frame, L = text tokens
        """
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B T) S D")

        # REVIEW ITEM 7: dimension mismatch #1. After flattening frames into the
        #     batch, x has batch B*T but the text only has batch B. Every frame of a
        #     clip follows the same instruction, so I copy the text once per frame.
        #     The padding mask has to be copied the same way, otherwise row i of the
        #     mask would describe a different sentence than row i of X.
        X = repeat(X, "B L D -> (B T) L D", T=T)
        if text_mask is not None:
            text_mask = repeat(text_mask, "B L -> (B T) L", T=T)

        # REVIEW ITEM 8: dimension mismatch #2 (S patches vs L words). My first fix
        #     zero-padded the shorter one so L == S. That isn't needed (attention is
        #     fine with S != L, see REVIEW ITEM 1) and it actively hurt:
        #       - zero keys aren't masked, so each gets score 0 and still takes
        #         softmax weight away from the real words
        #       - padding x adds fake *queries*, so the output is no longer length S
        #     It also had bugs: X.shape[2] is D (not L), JAX arrays have no .concat,
        #     and jnp.concatenate returns a new array instead of editing X in place.
        #     The real cause was Q coming from the text; with Q from the patches the
        #     output is (B*T, S, D) on its own.

        # REVIEW ITEM 9: after I added X to GroupedCrossAttention's signature, the
        #     positional arguments here were off by one: the text never got passed,
        #     `deterministic` landed in local_window_size, rngs in cache,
        #     return_weights in rngs, and static_argnums pointed at the wrong slots.
        #     Positions now line up with GroupedCrossAttention.__call__:
        #     x=0, X=1, text_mask=2, deterministic=3, rngs=4, return_weights=5.
        #     Python bools must be static under remat; arrays must not be.
        out, _, attn_weights = nnx.remat(
            self.attn,
            static_argnums=(3, 5),
        )(x, X, text_mask, deterministic, rngs, return_weights)
        out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
        if attn_weights is not None:
            # (B*T, N, S, L): how much each patch attends to each word
            attn_weights = rearrange(
                attn_weights, "(B T) N S L -> B T N S L", B=B, T=T
            )
        return out, None, attn_weights


class SpaceSelfAttention(nnx.Module):
    """Space self-attention with modality routing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
        self.attn = GroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
            qk_norm_type=qk_norm_type,
            rope_theta=rope_theta,
            is_causal=False,
            use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
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

        out, _, attn_weights = nnx.remat(
            self.attn,
            static_argnums=(2, 3, 4, 6),
        )(x, mask, None, deterministic, None, rngs, return_weights)
        out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
        if attn_weights is not None:
            attn_weights = rearrange(
                attn_weights, "(B T) N S1 S2 -> B T N S1 S2", B=B, T=T
            )
        return out, None, attn_weights


class TimeSelfAttention(nnx.Module):
    """Time self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
        self.attn = GroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
            qk_norm_type=qk_norm_type,
            rope_theta=rope_theta,
            is_causal=True,
            use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale,
            use_seq_parallel=use_seq_parallel,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
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
            mask = repeat(mask, "B ... -> (B S) ...", S=S)

        out, new_cache, attn_weights = nnx.remat(
            self.attn,
            static_argnums=(2, 3, 6),
        )(x, mask, local_window_size, deterministic, cache, rngs, return_weights)
        out = rearrange(out, "(B S) T D -> B T S D", B=B, S=S)
        if attn_weights is not None:
            attn_weights = rearrange(
                attn_weights, "(B S) N T1 T2 -> B S N T1 T2", B=B, S=S
            )
        return out, new_cache, attn_weights


class BlockCausalLayer(nnx.Module):
    """Single block-causal transformer layer (alternating space/time attention)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        mlp_ratio: float = 4.0,
        layer_index: int = 0,
        time_every: int = 4,
        time_layer_offset: int = 1,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
        mesh_rules: MeshRules,
    ):
        self.layer_index = layer_index
        self.time_every = time_every
        self.time_layer_offset = time_layer_offset
        param_dtype = to_jnp_dtype(param_dtype)

        self.norm = nnx.RMSNorm(
            dim,
            use_scale=use_rmsnorm_scale,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # Time or space attention
        is_time_layer = (
            self.layer_index + self.time_layer_offset
        ) % self.time_every == 0
        if is_time_layer:
            self.attn = TimeSelfAttention(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout_rate=dropout_rate,
                qk_norm_type=qk_norm_type,
                rope_theta=rope_theta,
                use_bias=use_bias,
                use_rmsnorm_scale=use_rmsnorm_scale,
                use_seq_parallel=use_seq_parallel,
                dtype=dtype,
                param_dtype=param_dtype,
                mesh_rules=mesh_rules,
                rngs=rngs,
            )
        else:
            # SpaceSelfAttention doesn't need seq_parallel - it operates on (B*T, S, D)
            # with T already in the batch dimension
            self.attn = SpaceSelfAttention(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout_rate=dropout_rate,
                qk_norm_type=qk_norm_type,
                rope_theta=rope_theta,
                use_bias=use_bias,
                use_rmsnorm_scale=use_rmsnorm_scale,
                dtype=dtype,
                param_dtype=param_dtype,
                mesh_rules=mesh_rules,
                rngs=rngs,
            )

        # MLP
        self.mlp = MLP(
            dim,
            mlp_ratio,
            dropout_rate,
            use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

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
        y, new_cache, attn_weights = self.attn(
            y,
            mask=attn_mask,
            local_window_size=local_window_size,
            deterministic=deterministic,
            cache=cache,
            rngs=rngs,
            return_weights=return_weights,
        )
        x = x + y

        # MLP
        y = nnx.remat(self.mlp)(x, deterministic=deterministic, rngs=rngs)
        x = x + y
        return x, new_cache, attn_weights


# REVIEW ITEM 11: I split the text-conditioned stack into three explicit layer
#     kinds instead of overloading "space" layers:
#       - "space":      patch <-> patch self-attention inside one frame
#       - "space_text": patch -> text cross-attention inside one frame
#       - "time":       causal attention across frames, per patch position
#     My first version replaced *every* space layer with cross-attention, so the
#     model lost patch-to-patch mixing entirely. Keeping the kinds separate lets the
#     transformer interleave them (see build_text_layer_schedule) and lets me read
#     the schedule off a tuple of strings when debugging.
LAYER_SPACE = "space"
LAYER_SPACE_TEXT = "space_text"
LAYER_TIME = "time"
LAYER_KINDS = (LAYER_SPACE, LAYER_SPACE_TEXT, LAYER_TIME)


class BlockCausalTextLayer(nnx.Module):
    """Single pre-norm block: one attention (space / space_text / time) + MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        layer_kind: str,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        use_text_gate: bool = True,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
        mesh_rules: MeshRules,
    ):
        if layer_kind not in LAYER_KINDS:
            raise ValueError(
                f"layer_kind must be one of {LAYER_KINDS}, got {layer_kind!r}"
            )
        self.layer_kind = layer_kind
        self.use_text_gate = use_text_gate
        param_dtype = to_jnp_dtype(param_dtype)

        self.norm = nnx.RMSNorm(
            dim,
            use_scale=use_rmsnorm_scale,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        attn_kwargs: dict[str, Any] = dict(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
            qk_norm_type=qk_norm_type,
            rope_theta=rope_theta,
            use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )
        if layer_kind == LAYER_TIME:
            self.attn = TimeSelfAttention(use_seq_parallel=use_seq_parallel, **attn_kwargs)
        elif layer_kind == LAYER_SPACE:
            # space attention runs on (B*T, S, D): T is already folded into the
            # batch, so it never needs sequence parallelism
            self.attn = SpaceSelfAttention(**attn_kwargs)
        else:
            self.attn = SpaceTextCrossAttention(**attn_kwargs)

            # REVIEW ITEM 12: pre-norm only normalizes the *query* stream (x). The
            #     text comes from a different network, so its scale has nothing to
            #     do with the video residual stream. I give every cross-attention
            #     layer its own RMSNorm for the text so K/V see well-scaled inputs
            #     (same idea as the qk-norm, applied before the projection).
            self.text_norm = nnx.RMSNorm(
                dim,
                use_scale=use_rmsnorm_scale,
                dtype=jnp.float32,
                param_dtype=param_dtype,
                rngs=rngs,
            )

            # REVIEW ITEM 13: a learnable gate, initialized to 0, multiplies the
            #     cross-attention output: x = x + tanh(gate) * xattn(x, text).
            #     (Flamingo does this.) At step 0 the layer is an exact no-op, so a
            #     pretrained video model isn't disrupted by randomly initialized
            #     cross-attention, and the model "opens" the gate only as fast as the
            #     text proves useful. tanh keeps it in [-1, 1].
            #     Side effect worth remembering: at init the gradient w.r.t. the
            #     cross-attention weights is exactly 0 (they're multiplied by
            #     tanh(0)); only the gate gets gradient on step 1, then everything
            #     else follows. A "dead params" test has to account for that.
            if use_text_gate:
                self.text_gate = nnx.Param(
                    jnp.zeros((1,), dtype=param_dtype),
                    sharding_names=(None,),  # one scalar per layer, no sharding
                )

        # MLP
        self.mlp = MLP(
            dim,
            mlp_ratio,
            dropout_rate,
            use_bias=use_bias,
            use_rmsnorm_scale=use_rmsnorm_scale,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

    @property
    def is_time_layer(self) -> bool:
        return self.layer_kind == LAYER_TIME

    @property
    def is_text_layer(self) -> bool:
        return self.layer_kind == LAYER_SPACE_TEXT

    def __call__(
        self,
        x,
        text=None,
        *,
        text_mask: jnp.ndarray | None = None,
        space_mask: jnp.ndarray | None = None,
        time_mask: jnp.ndarray | None = None,
        time_local_window_size: int | tuple[int, int] | None = None,
        deterministic: bool = True,
        cache: KVCache | None = None,
        rngs: nnx.Rngs | None = None,
        return_weights: bool = False,
    ):
        """
        x: (B, T, S, D) video tokens
        text: (B, L, D) text tokens, already projected to D (only used by space_text)
        text_mask: (B, L) padding mask, 1 = real word, 0 = pad
        """
        y = self.norm(x)

        # REVIEW ITEM 14: my old layer used one call for every attention type and
        #     passed (text, y) positionally, so video and text were swapped,
        #     TimeSelfAttention received text it doesn't accept, and cross-attention
        #     got space_mask + local_window_size it doesn't use. Each kind now gets
        #     exactly the arguments it takes. Only time layers carry a KV cache:
        #     space and cross-attention never look at other frames, so there's
        #     nothing to cache across steps.
        if self.layer_kind == LAYER_TIME:
            y, new_cache, attn_weights = self.attn(
                y,
                mask=time_mask,
                local_window_size=time_local_window_size,
                deterministic=deterministic,
                cache=cache,
                rngs=rngs,
                return_weights=return_weights,
            )
        elif self.layer_kind == LAYER_SPACE:
            y, new_cache, attn_weights = self.attn(
                y,
                mask=space_mask,
                deterministic=deterministic,
                rngs=rngs,
                return_weights=return_weights,
            )
        else:
            if text is None:
                raise ValueError("space_text layer called without text tokens")
            text_n = self.text_norm(text).astype(y.dtype)
            y, new_cache, attn_weights = self.attn(
                y,
                text_n,
                text_mask=text_mask,
                deterministic=deterministic,
                rngs=rngs,
                return_weights=return_weights,
            )
            if self.use_text_gate:
                y = jnp.tanh(self.text_gate.value[0]).astype(y.dtype) * y

        x = x + y

        # MLP
        y = nnx.remat(self.mlp)(x, deterministic=deterministic, rngs=rngs)
        x = x + y
        return x, new_cache, attn_weights


class BlockCausalTransformer(nnx.Module):
    """Stack of block-causal transformer layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        depth: int,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        time_layer_offset: int = 1,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        use_residual_lambdas: bool = False,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
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
                sharding_names=(None,),  # Small per-layer params, no sharding needed
            )
            self.x0_lambdas = nnx.Param(
                jnp.zeros(depth, dtype=param_dtype),
                sharding_names=(None,),  # Small per-layer params, no sharding needed
            )

        # Create layers
        self.layers = _nnx_list(
            [
                BlockCausalLayer(
                    dim=d_model,
                    num_heads=n_heads,
                    num_kv_heads=n_kv_heads,
                    dropout_rate=dropout_rate,
                    qk_norm_type=qk_norm_type,
                    mlp_ratio=mlp_ratio,
                    layer_index=i,
                    time_every=time_every,
                    time_layer_offset=time_layer_offset,
                    rope_theta=rope_theta,
                    use_bias=use_bias,
                    use_rmsnorm_scale=use_rmsnorm_scale,
                    use_seq_parallel=use_seq_parallel,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    mesh_rules=mesh_rules,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )

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
            cache_i = (
                caches.get(time_index) if caches is not None and is_time_layer else None
            )

            x, new_cache_i, w_i = layer(
                x,
                space_mask=space_mask,
                time_mask=time_mask,
                time_local_window_size=time_local_window_size,
                deterministic=deterministic,
                cache=cache_i,
                rngs=rngs,
                return_weights=return_weights,
            )
            if all_weights is not None:
                all_weights.append(w_i)

            if new_caches is not None and new_cache_i is not None:
                new_caches[time_index] = new_cache_i
            if is_time_layer:
                time_index += 1

        return x, new_caches, all_weights

    def estimate_attention_flops(
        self,
        batch_size: int,
        seq_time: int,
        seq_space: int,
        time_window: int | None = None,
    ) -> int:
        """Attention FLOPs per training step (forward + backward).

        Computes FLOPs for Q@K^T and attn@V operations only (not weight matrices).
        Factor of 12 = 2 matmuls × 2 ops × 3 (forward + backward).
        """
        n_time = sum(layer.is_time_layer for layer in self.layers)
        n_space = self.depth - n_time
        t_eff = seq_time if time_window is None else min(seq_time, time_window)
        space_attn = (
            12 * n_space * self.d_model * (seq_space**2) * batch_size * seq_time
        )
        time_attn = (
            12 * n_time * self.d_model * seq_time * t_eff * batch_size * seq_space
        )
        return int(space_attn + time_attn)

    def count_excluded_params(self) -> int:
        """Count params excluded from FLOP estimation (per-layer scalars)."""
        if not self.use_residual_lambdas:
            return 0
        return self.resid_lambdas.value.size + self.x0_lambdas.value.size


def build_text_layer_schedule(
    depth: int,
    time_every: int = 4,
    time_layer_offset: int = 1,
    text_every: int = 4,
    text_layer_offset: int = 3,
) -> tuple[str, ...]:
    """Decide the kind of every layer in a BlockCausalTextTransformer.

    REVIEW ITEM 15: I keep the original time-layer rule untouched
        ((i + time_layer_offset) % time_every == 0) so the time layers sit at the
        same depths as in BlockCausalTransformer, which matters if I ever load its
        weights. Among the remaining (space) slots, every text_every-th one becomes
        a space_text layer. With the defaults and depth 8 that gives:

            0 space | 1 space_text | 2 space | 3 time |
            4 space | 5 space_text | 6 space | 7 time

        so each group of four reads the frame, reads the instruction, mixes the two
        within the frame, then looks across time.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if time_every <= 0 or text_every <= 0:
        raise ValueError("time_every and text_every must be positive")

    schedule = []
    for i in range(depth):
        if (i + time_layer_offset) % time_every == 0:
            schedule.append(LAYER_TIME)
        elif (i + text_layer_offset) % text_every == 0:
            schedule.append(LAYER_SPACE_TEXT)
        else:
            schedule.append(LAYER_SPACE)
    return tuple(schedule)


class BlockCausalTextTransformer(nnx.Module):
    """Stack of space / space_text / time layers: a text-conditioned video transformer.

    Same contract as BlockCausalTransformer (input and output are (B, T, S, D), KV
    caches only for time layers), plus a text stream that every space_text layer
    cross-attends to.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        depth: int,
        text_dim: int | None = None,
        layer_schedule: Sequence[str] | None = None,
        dropout_rate: float = 0.0,
        qk_norm_type: str | None = None,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        time_layer_offset: int = 1,
        text_every: int = 4,
        text_layer_offset: int = 3,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        use_rmsnorm_scale: bool = True,
        use_seq_parallel: bool = False,
        use_text_gate: bool = True,
        dtype: Any = jnp.float32,
        param_dtype: Any = jnp.float32,
        use_residual_lambdas: bool = False,
        *,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.depth = depth
        self.use_residual_lambdas = use_residual_lambdas
        self.dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)

        if layer_schedule is None:
            layer_schedule = build_text_layer_schedule(
                depth, time_every, time_layer_offset, text_every, text_layer_offset
            )
        layer_schedule = tuple(layer_schedule)
        if len(layer_schedule) != depth:
            raise ValueError(
                f"layer_schedule has {len(layer_schedule)} entries, depth is {depth}"
            )
        bad = [k for k in layer_schedule if k not in LAYER_KINDS]
        if bad:
            raise ValueError(f"unknown layer kinds {bad}; expected {LAYER_KINDS}")
        # A stack with no space_text layer would silently ignore the text.
        if LAYER_SPACE_TEXT not in layer_schedule:
            raise ValueError("layer_schedule has no space_text layer; text would be unused")
        self.layer_schedule = layer_schedule

        # REVIEW ITEM 16: my text encoder's width (d_model=512 in
        #     TextEncoderConfig) doesn't have to match the video model's d_model.
        #     Instead of making every cross-attention layer handle a second input
        #     width, I project the text to d_model once here and share the result
        #     across all space_text layers. When the widths already match I skip
        #     the projection entirely.
        if text_dim is not None and text_dim != d_model:
            self.text_proj = nnx.Linear(
                text_dim,
                d_model,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), mesh_rules("mlp")
                ),
                rngs=rngs,
            )
        else:
            self.text_proj = None

        if use_residual_lambdas:
            self.resid_lambdas = nnx.Param(
                jnp.ones(depth, dtype=param_dtype),
                sharding_names=(None,),
            )
            self.x0_lambdas = nnx.Param(
                jnp.zeros(depth, dtype=param_dtype),
                sharding_names=(None,),
            )

        self.layers = _nnx_list(
            [
                BlockCausalTextLayer(
                    dim=d_model,
                    num_heads=n_heads,
                    num_kv_heads=n_kv_heads,
                    layer_kind=kind,
                    dropout_rate=dropout_rate,
                    qk_norm_type=qk_norm_type,
                    mlp_ratio=mlp_ratio,
                    rope_theta=rope_theta,
                    use_bias=use_bias,
                    use_rmsnorm_scale=use_rmsnorm_scale,
                    use_seq_parallel=use_seq_parallel,
                    use_text_gate=use_text_gate,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    mesh_rules=mesh_rules,
                    rngs=rngs,
                )
                for kind in layer_schedule
            ]
        )

    def __call__(
        self,
        x,
        text,
        *,
        text_mask: jnp.ndarray | None = None,
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
            x: (B, T, S, D) video tokens
            text: (B, L, text_dim) text tokens from the text encoder
            text_mask: (B, L) tokenizer attention_mask (1 = real word, 0 = pad)
            space_mask, time_mask, time_local_window_size, deterministic, caches,
            rngs, return_weights: same as BlockCausalTransformer

        Returns:
            (x, new_caches, all_weights). all_weights is None unless return_weights;
            for space_text layers the entry is (B, T, N, S, L) patch->word weights.
        """
        # REVIEW ITEM 17: shapes are static under jit, so I can check them here in
        #     plain Python and fail with a readable message instead of an einops
        #     error three calls deeper. What I can NOT check cheaply under jit is
        #     the mask *content*: every sample needs at least one real token
        #     (BOS/CLS counts), otherwise its softmax row has no valid keys and the
        #     output is NaN or meaningless.
        if x.ndim != 4:
            raise ValueError(f"x must be (B, T, S, D), got shape {x.shape}")
        if text.ndim != 3:
            raise ValueError(f"text must be (B, L, D_text), got shape {text.shape}")
        if text.shape[0] != x.shape[0]:
            raise ValueError(
                f"batch mismatch: x has B={x.shape[0]}, text has B={text.shape[0]}"
            )
        if text_mask is not None and text_mask.shape != text.shape[:2]:
            raise ValueError(
                f"text_mask must be (B, L)={text.shape[:2]}, got {text_mask.shape}"
            )

        if self.text_proj is not None:
            text = self.text_proj(text)
        text = text.astype(x.dtype)

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
            # caches are indexed by time-layer count, matching create_transformer_caches
            cache_i = (
                caches.get(time_index) if caches is not None and is_time_layer else None
            )

            x, new_cache_i, w_i = layer(
                x,
                text,
                text_mask=text_mask,
                space_mask=space_mask,
                time_mask=time_mask,
                time_local_window_size=time_local_window_size,
                deterministic=deterministic,
                cache=cache_i,
                rngs=rngs,
                return_weights=return_weights,
            )
            if all_weights is not None:
                all_weights.append(w_i)

            if new_caches is not None and new_cache_i is not None:
                new_caches[time_index] = new_cache_i
            if is_time_layer:
                time_index += 1

        return x, new_caches, all_weights

    def estimate_attention_flops(
        self,
        batch_size: int,
        seq_time: int,
        seq_space: int,
        seq_text: int,
        time_window: int | None = None,
    ) -> int:
        """Attention FLOPs per training step (forward + backward).

        REVIEW ITEM 18: cross-attention cost is S x L per frame instead of S x S,
            so with ~a dozen words against hundreds of patches a space_text layer is
            much cheaper than a space layer. Factor 12 = 2 matmuls x 2 ops x 3
            (forward + backward), same convention as BlockCausalTransformer.
        """
        n_time = sum(layer.is_time_layer for layer in self.layers)
        n_text = sum(layer.is_text_layer for layer in self.layers)
        n_space = self.depth - n_time - n_text
        t_eff = seq_time if time_window is None else min(seq_time, time_window)
        bt = batch_size * seq_time
        space_attn = 12 * n_space * self.d_model * (seq_space**2) * bt
        text_attn = 12 * n_text * self.d_model * seq_space * seq_text * bt
        time_attn = (
            12 * n_time * self.d_model * seq_time * t_eff * batch_size * seq_space
        )
        return int(space_attn + text_attn + time_attn)

    def count_excluded_params(self) -> int:
        """Count params excluded from FLOP estimation (per-layer scalars)."""
        n = 0
        if self.use_residual_lambdas:
            n += self.resid_lambdas.value.size + self.x0_lambdas.value.size
        n += sum(
            layer.text_gate.value.size
            for layer in self.layers
            if layer.is_text_layer and layer.use_text_gate
        )
        return n


# ============================================================================
# Tokenizer
# ============================================================================

from transformers import AutoTokenizer

import jax.numpy as np
from flax import linen as nn


class TextEncoderEmbed(nn.Module):
    """Text Embedder only"""

    cfg: TextEncoderConfig

    @nn.compact
    def __call__(self, input_ids):
        _, seq_len = input_ids.shape
        assert seq_len <= self.cfg.max_seq_len

        # token ids -> vectors
        token_embed = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.d_model,
            name="token_embedding",
        )

        position_embed = nn.Embed(
            num_embeddings=self.cfg.max_seq_len,
            features=self.d_model,
            name="position_embeddng",
        )

        x = token_embed(input_ids)
        positions = jnp.arange(seq_len)
        x = x + position_embed(positions)

        return x


class TextEncoderPipeline(nnx.Module):
    """A UML-large-5 tmbedding model"""

    def __init__(self, cfg: TextEncoderConfig):
        self.d_model = cfg.d_model
        self.max_seq_len = cfg.max_seq_len
        self.tokenizer_name = "google-t5/t5-small"
        tokenizer = None
        self.embedding_model = TextEncoderEmbed(TextEncoderConfig(d_model=512))
        try:
            # Do NOT access the network
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                local_files_only=True,
            )
        except OSError:
            # Not cached → download and cache it
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer = tokenizer

    @nn.compact
    def __call__(self, text: list[str]):
        input_ids = self.tokenizer(text)
        x = self.embedding_model(input_ids)
        return jnp.array(x)


class TextEncoder(nnx.Module):
    """Text+video encoder"""

    def __init__(
        self, cfg: TextEncoderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs
    ):
        self.cfg = cfg
        self.n_latents = cfg.n_latents
        self.patch_size = cfg.patch_size
        self.context_length = cfg.context_length
        self.dataset_mean = cfg.dataset_mean
        self.dataset_std = cfg.dataset_std
        self.context_length = cfg.context_length
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)
        self.text_pipeline = TextEncoderEmbed(self.cfg.text_encoder_cfg)

        self.patch_proj = nnx.Linear(
            cfg.patch_size * cfg.patch_size * 3,
            cfg.d_model,
            use_bias=cfg.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )
        self.bottleneck_proj = nnx.Linear(
            cfg.d_model,
            cfg.d_bottleneck,
            use_bias=cfg.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            depth=cfg.depth,
            dropout_rate=cfg.dropout_rate,
            qk_norm_type=cfg.qk_norm_type,
            mlp_ratio=4.0,
            time_every=cfg.time_every,
            time_layer_offset=getattr(cfg, "time_layer_offset", 1),
            rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias,
            use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, "use_seq_parallel", False),
            dtype=dtype,
            param_dtype=param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

        self.mask_and_replace = MAEReplacer(
            D=cfg.d_model,
            p_min=cfg.mae_p_min,
            p_max=cfg.mae_p_max,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )
        self.latents_enc = nnx.Param(
            jax.random.normal(
                rngs.params(), (cfg.n_latents, cfg.d_model), dtype=param_dtype
            )
            * 0.02,
            sharding_names=mesh_rules("embed"),
        )
        self.tokenizer = TextEncoderPipeline(TextEncoderConfig(d_model=512))

    def get_token_layout(self, H: int, W: int) -> TokenLayout:
        n_patches = (H // self.patch_size) * (W // self.patch_size)
        return TokenLayout(
            ((Modality.LATENT, self.n_latents), (Modality.IMAGE, n_patches))
        )

    def create_static_caches(
        self,
        batch_size: int,
        H: int,
        W: int,
        window_size: int = 1024,
        dtype=jnp.float32,
    ) -> KVCachesDict:
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

    def __call__(
        self,
        videos,
        input_ids,
        *,
        deterministic: bool = True,
        caches: KVCachesDict | None = None,
        rngs: nnx.Rngs | None = None,
        mae_p_max: jnp.ndarray | None = None,
    ):
        # Videos in the [0, 255] range
        B, T, H, W, C = videos.shape
        if B != len(input_ids):
            return

        embedded_text = self.text_pipeline(input_ids)
        # Text shape
        BT, L, n, d = embedded_text.shape

        normalized_videos = normalize_with_dataset_stats(
            videos, mean=self.dataset_mean, std=self.dataset_std
        )

        patch_tokens = patchify(normalized_videos, patch=self.patch_size)
        proj_patches = self.patch_proj(patch_tokens)  # (B, T, Np, D)

        if deterministic or rngs is None:
            # apply causal masking durin inference
            proj_patches_masked = proj_patches
            patch_mask = jnp.zeros((B, T, proj_patches.shape[2], 1), dtype=jnp.bool_)
            keep_prob = jnp.ones((B, T, 1))
        else:
            proj_patches_masked, patch_mask, keep_prob = self.mask_and_replace(
                proj_patches, rngs=rngs, p_max_override=mae_p_max
            )

        # patch_mask is (B,T,Np,1), need to expand to pixels (B, T, Np, P*P)
        patch_mask_expanded = jnp.repeat(patch_mask, self.patch_size**2, axis=-1)
        frame_mask = unpatchify(patch_mask_expanded, self.patch_size, H, W)

        # Prepend learned latents
        latents = repeat(
            self.latents_enc.value.astype(proj_patches.dtype),
            "... -> b t ...",
            b=B,
            t=T,
        )
        text_tokens = repeat(
            embedded_text.astype(proj_patches.dtype),
            "... -> b t ...",
            b=B,
            t=T,
        )
        tokens = jnp.concatenate(
            [latents, proj_patches_masked], axis=2
        )  # (B, T, S=(Np+Nl+L), D)

        layout = self.get_token_layout(H, W)
        space_mask = layout.build_space_mask("encoder")  # (1, 1, q_len, k_len)

        # Feed tokens into transformer
        time_local_window_size = (
            (self.context_length - 1, 0) if self.context_length is not None else None
        )
        # REVIEW ITEM 19 (open): this still calls the plain BlockCausalTransformer,
        #     which takes one positional input, so passing text here raises. To wire
        #     text in I need to:
        #       1. build self.transformer as BlockCausalTextTransformer(text_dim=...)
        #       2. keep text as (B, L, D_text) -- NOT repeated over T here, the
        #          cross-attention repeats it per frame (REVIEW ITEM 7)
        #       3. pass the tokenizer's attention_mask as text_mask (REVIEW ITEM 4)
        #       4. port TextEncoderEmbed from flax.linen to nnx; linen modules
        #          can't hold state inside an nnx.Module this way
        encoded_tokens, new_caches, _ = self.transformer(
            text_tokens,
            tokens,
            space_mask=space_mask,
            time_local_window_size=time_local_window_size,
            deterministic=deterministic,
            caches=caches,
            rngs=rngs,
        )

        # Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, : self.n_latents]
        proj_tokens = nnx.tanh(self.bottleneck_proj(latent_tokens))

        # new_caches is None unless a KV cache was provided.
        return proj_tokens, (frame_mask, keep_prob), new_caches


class Encoder(nnx.Module):
    """Vision encoder with MAE masking."""

    def __init__(
        self, cfg: EncoderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs
    ):
        self.cfg = cfg
        self.n_latents = cfg.n_latents
        self.patch_size = cfg.patch_size
        self.context_length = cfg.context_length
        self.dataset_mean = cfg.dataset_mean
        self.dataset_std = cfg.dataset_std
        self.context_length = cfg.context_length
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)

        self.patch_proj = nnx.Linear(
            cfg.patch_size * cfg.patch_size * 3,
            cfg.d_model,
            use_bias=cfg.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )
        self.bottleneck_proj = nnx.Linear(
            cfg.d_model,
            cfg.d_bottleneck,
            use_bias=cfg.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            depth=cfg.depth,
            dropout_rate=cfg.dropout_rate,
            qk_norm_type=cfg.qk_norm_type,
            mlp_ratio=4.0,
            time_every=cfg.time_every,
            time_layer_offset=getattr(cfg, "time_layer_offset", 1),
            rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias,
            use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, "use_seq_parallel", False),
            dtype=dtype,
            param_dtype=param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

        self.mask_and_replace = MAEReplacer(
            D=cfg.d_model,
            p_min=cfg.mae_p_min,
            p_max=cfg.mae_p_max,
            dtype=dtype,
            param_dtype=param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )
        self.latents_enc = nnx.Param(
            jax.random.normal(
                rngs.params(), (cfg.n_latents, cfg.d_model), dtype=param_dtype
            )
            * 0.02,
            sharding_names=mesh_rules("embed"),
        )

    def get_token_layout(self, H: int, W: int) -> TokenLayout:
        n_patches = (H // self.patch_size) * (W // self.patch_size)
        return TokenLayout(
            ((Modality.LATENT, self.n_latents), (Modality.IMAGE, n_patches))
        )

    def create_static_caches(
        self,
        batch_size: int,
        H: int,
        W: int,
        window_size: int = 1024,
        dtype=jnp.float32,
    ) -> KVCachesDict:
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

    def __call__(
        self,
        videos,
        *,
        deterministic: bool = True,
        caches: KVCachesDict | None = None,
        rngs: nnx.Rngs | None = None,
        mae_p_max: jnp.ndarray | None = None,
    ):
        # Videos in the [0, 255] range
        B, T, H, W, C = videos.shape

        normalized_videos = normalize_with_dataset_stats(
            videos, mean=self.dataset_mean, std=self.dataset_std
        )
        patch_tokens = patchify(normalized_videos, patch=self.patch_size)
        proj_patches = self.patch_proj(patch_tokens)  # (B, T, Np, D)

        # MAE mask-and-replace on patch tokens (encoder input only during training)
        if deterministic or rngs is None:
            # Skip MAE masking during inference
            proj_patches_masked = proj_patches
            patch_mask = jnp.zeros((B, T, proj_patches.shape[2], 1), dtype=jnp.bool_)
            keep_prob = jnp.ones((B, T, 1))
        else:
            proj_patches_masked, patch_mask, keep_prob = self.mask_and_replace(
                proj_patches, rngs=rngs, p_max_override=mae_p_max
            )

        # patch_mask is (B,T,Np,1), need to expand to pixels (B, T, Np, P*P)
        patch_mask_expanded = jnp.repeat(patch_mask, self.patch_size**2, axis=-1)
        frame_mask = unpatchify(patch_mask_expanded, self.patch_size, H, W)

        # Prepend learned latents
        latents = repeat(
            self.latents_enc.value.astype(proj_patches.dtype),
            "... -> b t ...",
            b=B,
            t=T,
        )
        tokens = jnp.concatenate(
            [latents, proj_patches_masked], axis=2
        )  # (B, T, S=(Np+Nl), D)

        layout = self.get_token_layout(H, W)
        space_mask = layout.build_space_mask("encoder")  # (1, 1, q_len, k_len)

        # Feed tokens into transformer
        time_local_window_size = (
            (self.context_length - 1, 0) if self.context_length is not None else None
        )
        encoded_tokens, new_caches, _ = self.transformer(
            tokens,
            space_mask=space_mask,
            time_local_window_size=time_local_window_size,
            deterministic=deterministic,
            caches=caches,
            rngs=rngs,
        )

        # Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, : self.n_latents]
        proj_tokens = nnx.tanh(self.bottleneck_proj(latent_tokens))

        # new_caches is None unless a KV cache was provided.
        return proj_tokens, (frame_mask, keep_prob), new_caches


class Decoder(nnx.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.
    """

    def __init__(
        self, cfg: DecoderModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs
    ):
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
        self.up_proj = nnx.Linear(
            cfg.d_bottleneck,
            cfg.d_model,
            use_bias=cfg.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )
        self.patch_head = nnx.Linear(
            cfg.d_model,
            cfg.d_patch,
            use_bias=cfg.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros, mesh_rules("mlp")
            ),
            rngs=rngs,
        )

        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            depth=cfg.depth,
            dropout_rate=cfg.dropout_rate,
            qk_norm_type=cfg.qk_norm_type,
            mlp_ratio=4.0,
            time_every=cfg.time_every,
            time_layer_offset=getattr(cfg, "time_layer_offset", 1),
            rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias,
            use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, "use_seq_parallel", False),
            dtype=dtype,
            param_dtype=param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

        self.patch_queries = nnx.Param(
            jax.random.normal(
                rngs.params(), (self.n_patches, cfg.d_model), dtype=param_dtype
            )
            * 0.02,
            sharding_names=mesh_rules("embed"),
        )

    def get_token_layout(self) -> TokenLayout:
        return TokenLayout(
            ((Modality.LATENT, self.n_latents), (Modality.IMAGE, self.n_patches))
        )

    def create_static_caches(
        self, batch_size: int, window_size: int = 1024, dtype=jnp.float32
    ) -> KVCachesDict:
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
        rngs: nnx.Rngs | None = None,
    ):
        # Always expect unpacked input: (B, T, n_latents, d_bottleneck)
        B, T, N_l, d_bottleneck = z.shape

        # Up-project latent bottleneck to d_model (per latent token)
        latents = self.up_proj(z)  # (B, T, N_l, D)

        # Learned per-patch query tokens (owned by the decoder)
        patches = repeat(
            self.patch_queries.value.astype(latents.dtype), " ... -> b t ...", b=B, t=T
        )  # (B, T, Np, D)

        # Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=-2)

        # Make mask
        layout = self.get_token_layout()
        space_mask = layout.build_space_mask("decoder")

        time_local_window_size = (
            (self.context_length - 1, 0) if self.context_length is not None else None
        )
        x, new_caches, _ = self.transformer(
            tokens,
            space_mask=space_mask,
            time_local_window_size=time_local_window_size,
            deterministic=deterministic,
            caches=caches,
            rngs=rngs,
        )

        # Prediction head over the patch-query slice
        x_patches = x[:, :, N_l:, :]  # (B, T, Np, D)
        pred_btnd = self.patch_head(x_patches)  # (B, T, Np, D_patch)
        out_normalized_frames = unpatchify(
            pred_btnd, patch=self.patch_size, H=self.H, W=self.W
        )
        out_frames = unnormalize_with_dataset_stats(
            out_normalized_frames, mean=self.dataset_mean, std=self.dataset_std
        )
        return out_frames, new_caches


class Tokenizer(nnx.Module):
    """Complete tokenizer (encoder + decoder)."""

    def __init__(
        self, cfg: TokenizerModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs
    ):
        self.cfg = cfg

        # Create encoder and decoder
        self.encoder = Encoder(cfg.encoder, mesh_rules=mesh_rules, rngs=rngs)
        self.decoder = Decoder(cfg.decoder, mesh_rules=mesh_rules, rngs=rngs)

    def __call__(
        self,
        videos,
        *,
        deterministic: bool = True,
        caches: TokenizerCaches | None = None,
        rngs: nnx.Rngs | None = None,
        mae_p_max: jnp.ndarray | None = None,
    ):
        encoder_caches = caches.encoder if caches is not None else None
        decoder_caches = caches.decoder if caches is not None else None
        z, aux, encoder_caches = self.encoder(
            videos,
            deterministic=deterministic,
            caches=encoder_caches,
            rngs=rngs,
            mae_p_max=mae_p_max,
        )
        recon, decoder_caches = self.decoder(
            z, deterministic=deterministic, caches=decoder_caches, rngs=rngs
        )
        return (
            recon,
            aux,
            TokenizerCaches(encoder=encoder_caches, decoder=decoder_caches),
        )

    def encode(
        self,
        videos,
        *,
        deterministic: bool = True,
        caches: KVCachesDict | None = None,
        rngs: nnx.Rngs | None = None,
        mae_p_max: jnp.ndarray | None = None,
    ):
        # Always returns unpacked: (B, T, n_latents, d_bottleneck)
        return self.encoder(
            videos,
            deterministic=deterministic,
            caches=caches,
            rngs=rngs,
            mae_p_max=mae_p_max,
        )

    def decode(
        self,
        z,
        *,
        deterministic: bool = True,
        caches: KVCachesDict | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        # Always expects unpacked: (B, T, n_latents, d_bottleneck)
        return self.decoder(z, deterministic=deterministic, caches=caches, rngs=rngs)

    def create_static_caches(
        self,
        batch_size: int,
        H: int,
        W: int,
        window_size: int = 1024,
        dtype=jnp.float32,
    ) -> TokenizerCaches:
        """Creates namespaced encoder and decoder KV cache buffers."""
        encoder_window_size = (
            self.encoder.context_length
            if self.encoder.context_length is not None
            else window_size
        )

        return TokenizerCaches(
            encoder=self.encoder.create_static_caches(
                batch_size=batch_size,
                H=H,
                W=W,
                window_size=encoder_window_size,
                dtype=dtype,
            ),
            decoder=self.decoder.create_static_caches(
                batch_size=batch_size, window_size=window_size, dtype=dtype
            ),
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

        Follows https://github.com/karpathy/nanochat/discussions/420.
        """
        S = self.decoder.get_token_layout().S

        # Weight FLOPs (excluding embeddings/scalars)
        total_params = self.num_scaling_params()
        excluded = self.count_excluded_params()
        weight_flops = 6 * (total_params - excluded) * batch_size * seq_length * S

        # Attention FLOPs
        enc_attn = self.encoder.transformer.estimate_attention_flops(
            batch_size, seq_length, S
        )
        dec_attn = self.decoder.transformer.estimate_attention_flops(
            batch_size, seq_length, S
        )

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
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        dtype = to_jnp_dtype(dtype)
        param_dtype = to_jnp_dtype(param_dtype)
        self.dtype = dtype

        # Base "action token" embedding (used always)
        self.base_action_emb = nnx.Param(
            jax.random.normal(rngs.params(), (d_model,), dtype=param_dtype) * 0.02,
            sharding_names=mesh_rules("embed"),
        )

        # Embed binary actions
        if num_binary_actions > 0:
            binary_embeds = [
                nnx.Embed(
                    2,
                    d_model,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    embedding_init=nnx.with_partitioning(
                        nnx.initializers.normal(stddev=1.0), mesh_rules("embed")
                    ),
                    rngs=rngs,
                )
                for _ in range(num_binary_actions)
            ]
            self.binary_embeds_list = _nnx_list(
                binary_embeds
            )  # num_binary_actions * (B, T, d_model)
        else:
            self.binary_embeds_list = None

        # Embed categorical action
        if categorical_action_dim > 0:
            self.categorical_embeds = nnx.Embed(
                categorical_action_dim,
                d_model,
                dtype=dtype,
                param_dtype=param_dtype,
                embedding_init=nnx.with_partitioning(
                    nnx.initializers.normal(stddev=1.0), mesh_rules("embed")
                ),
                rngs=rngs,
            )  # (B, T, d_model)
        else:
            self.categorical_embeds = None

        # Continuous actions: linear projection
        if continuous_action_dim > 0:
            self.continuous_proj = nnx.Linear(
                continuous_action_dim,
                d_model,
                use_bias=use_bias,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), mesh_rules("mlp")
                ),
                rngs=rngs,
            )  # (B, T, d_model)
        else:
            self.continuous_proj = None

    def __call__(
        self,
        actions: Actions,
        batch_time_shape: Tuple[int, int],
        as_tokens: bool = True,
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
            freq_dim,
            out_dim,
            use_bias=True,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(0.02), mesh_rules("mlp")
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            out_dim,
            out_dim,
            use_bias=True,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(0.02), mesh_rules("mlp")
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, t: jax.Array) -> jax.Array:
        """Embed scalar timestep(s).

        Args:
            t: (...) float array of timestep values.

        Returns:
            (..., out_dim) embedding array.
        """
        t = (
            t.astype(jnp.float32) * 1000.0
        )  # scale so t spans a useful range of the sinusoidal basis (DiT convention)
        half = self.freq_dim // 2
        freqs = jnp.exp(
            -math.log(self.max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        args = t[..., None] * freqs  # (..., half)
        emb = jnp.concatenate(
            [jnp.cos(args), jnp.sin(args)], axis=-1
        )  # (..., freq_dim)
        emb = emb.astype(self.dtype)
        emb = nnx.silu(self.dense1(emb))
        return self.dense2(emb)


class Dynamics(nnx.Module):
    """Dynamics model (world model)."""

    def __init__(
        self, cfg: DynamicsModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs
    ):
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
        self.spatial_proj = nnx.Linear(
            cfg.d_bottleneck * cfg.packing_factor,
            cfg.d_model,
            use_bias=cfg.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), mesh_rules("mlp")
            ),
            rngs=rngs,
        )

        # Register tokens
        self.register_tokens = nnx.Param(
            jax.random.normal(
                rngs.params(), (cfg.n_register, cfg.d_model), dtype=self.param_dtype
            )
            * 0.02,
            sharding_names=mesh_rules("embed"),
        )

        # Action encoder
        self.action_encoder = ActionEncoder(
            d_model=cfg.d_model,
            num_binary_actions=cfg.num_binary_actions,
            categorical_action_dim=cfg.categorical_action_dim,
            continuous_action_dim=cfg.continuous_action_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

        # Transformer
        self.transformer = BlockCausalTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            depth=cfg.depth,
            dropout_rate=cfg.dropout_rate,
            qk_norm_type=cfg.qk_norm_type,
            mlp_ratio=cfg.mlp_ratio,
            time_every=cfg.time_every,
            time_layer_offset=cfg.time_layer_offset,
            rope_theta=cfg.rope_theta,
            use_bias=cfg.use_bias,
            use_rmsnorm_scale=cfg.use_rmsnorm_scale,
            use_seq_parallel=getattr(cfg, "use_seq_parallel", False),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_residual_lambdas=cfg.use_residual_lambdas,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

        # Sinusoidal embeddings for shortcut conditioning
        # step_idx ∈ {0,...,log2(k_max)} passed as float; tau ∈ [0,1] (tau_idx/k_max)
        half_dim = cfg.d_model // 2
        self.step_embed = TimestepEmbedder(
            out_dim=half_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )
        self.signal_embed = TimestepEmbedder(
            out_dim=half_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh_rules=mesh_rules,
            rngs=rngs,
        )

        # Output head (zero-init)
        self.flow_x_head = nnx.Linear(
            cfg.d_model,
            cfg.d_bottleneck * cfg.packing_factor,
            use_bias=cfg.use_bias,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros, mesh_rules("mlp")
            ),
            bias_init=nnx.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
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

    def create_static_caches(
        self,
        batch_size: int,
        n_latents: int,
        window_size: int = 1024,
        n_agent: int = 0,
        dtype=jnp.float32,
    ) -> KVCachesDict:
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
            dtype=dtype,
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
        rngs: nnx.Rngs | None = None,
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
        packed_enc_tokens = rearrange(
            unpacked_enc_tokens, "b t (n p) d -> b t n (p d)", p=self.packing_factor
        )
        # Project spatial tokens to d_model
        spatial_tokens = self.spatial_proj(
            packed_enc_tokens
        )  # (B, T, n_spatial, d_model)

        # Encode actions to d_model
        action_token = self.action_encoder(
            actions=actions,
            batch_time_shape=(B, T),
        )  # (B, T, 1, d_model)

        # Prepare learned register tokens
        B, T = spatial_tokens.shape[:2]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value.astype(self.dtype)[
                None, None, ...
            ],  # (1, 1 ,n_register, d_model)
            (B, T, self.n_register, self.d_model),
        )

        # Shortcut embeddings (sinusoidal, concatenated to single token)
        # step_indices: int log2(K) → float for sinusoidal PE
        # tau_indices: int τ*k_max → normalize to [0,1] for sinusoidal PE
        step_emb = self.step_embed(
            step_indices.astype(jnp.float32)
        )  # (B, T, d_model//2)
        signal_emb = self.signal_embed(
            tau_indices.astype(jnp.float32) / self.k_max
        )  # (B, T, d_model//2)
        shortcut_token = jnp.concatenate([step_emb, signal_emb], axis=-1)[
            :, :, None, :
        ]  # (B, T, 1, d_model)

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
            tokens,
            space_mask=space_mask,
            time_mask=time_mask,
            time_local_window_size=time_local_window_size,
            deterministic=deterministic,
            caches=caches,
            rngs=rngs,
        )

        spatial_tokens = x[:, :, layout.slices()[Modality.SPATIAL], :]
        x1_hat_packed = self.flow_x_head(spatial_tokens)  # (B, T, n_spatial, d_spatial)

        # Unpack before returning
        x1_hat = rearrange(
            x1_hat_packed, "b t n (p d) -> b t (n p) d", p=self.packing_factor
        )

        h_t = (
            x[:, :, layout.slices()[Modality.AGENT], :]
            if task_embeddings is not None
            else None
        )  # (B,T,n_agent,D) or None
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
        attn_flops = self.transformer.estimate_attention_flops(
            batch_size, seq_length, S, self.cfg.context_length
        )

        return int(weight_flops + attn_flops)
