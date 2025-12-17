from functools import lru_cache
from dataclasses import asdict

import jax.numpy as jnp
import flax.linen as nn
import jax
import time
from flax.core import FrozenDict, freeze, unfreeze
import flax
from typing import Optional, Tuple, Any, Dict
from einops import rearrange, repeat
import math
import orbax.checkpoint as ocp
from .utils import Modality, TokenLayout, make_manager, from_dict, recursive_list_to_tuple
from .data import patchify, unpatchify
from .configs import TokenizerConfig, DynamicsModelConfig



@flax.struct.dataclass
class KVCache:
    """
    Ring buffer KV cache for JIT compilation.
    """
    k: jnp.ndarray      # (B, Max_T, K, H)
    v: jnp.ndarray      # (B, Max_T, K, H)
    index: jnp.ndarray  # scalar integer (i32)
    window_size: int = flax.struct.field(pytree_node=False)

    @classmethod
    def init(cls, batch_size, window_size, num_kv_heads, head_dim, dtype=jnp.float32):
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
        
        return self.replace(k=k_updated, v=v_updated, index=self.index + T)

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



class RotaryEmbedding1D(nn.Module):
    dim: int
    theta: float = 10000.0
    dtype: Any = jnp.float32

    def setup(self):
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        self.inv_freq = self.variable('constants', 'inv_freq', lambda: inv_freq)

    def __call__(self, q, k, start_pos=0):
        """
        q: (B, T, N, H)
        k: (B, T, K, H)
        start_pos: scalar int (the absolute position of the first frame in T)
        """
        T = q.shape[1]
        t_indices = jnp.arange(T, dtype=self.dtype) + start_pos
        freqs = jnp.outer(t_indices, self.inv_freq.value) # (T, dim//2)

        freqs_cos = jnp.cos(freqs)
        freqs_sin = jnp.sin(freqs)

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


class MAEReplacer(nn.Module):
    p_min: float = 0.0
    p_max: float = 0.9

    @nn.compact
    def __call__(self, patches_btnd: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # patches_btnd: (B,T,Np,D)
        B, T, Np, D = patches_btnd.shape
        mask_token = self.param("mask_token", nn.initializers.normal(0.02), (D,))
        # draw RNGs from a named stream
        rng = self.make_rng("mae")
        p_rng, m_rng = jax.random.split(rng)
        p_bt = jax.random.uniform(p_rng, (B, T), minval=self.p_min, maxval=self.p_max)  # (B,T)
        keep_prob_bt1 = 1.0 - p_bt[..., None]                                           # (B,T,1)
        keep = jax.random.bernoulli(m_rng, keep_prob_bt1, (B, T, Np))                   # (B,T,Np)
        keep = keep[..., None]                                                          # (B,T,Np,1)
        replaced = jnp.where(keep, patches_btnd, mask_token.reshape(1, 1, 1, D))
        mae_mask = (~keep).astype(jnp.bool_)                                            # (B,T,Np,1)
        return replaced, mae_mask, keep_prob_bt1


# ---------- small building blocks ----------

class RMSNorm(nn.Module):
    eps: float = 1e-6
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * (scale / jnp.sqrt(var + self.eps))

class MLP(nn.Module):
    """
    Transformer MLP with optional SwiGLU gating.

    Args:
      d_model:          input/output width of the block (last-dim of x).
      mlp_ratio:    expansion factor for hidden size if NOT using 2/3 parity.
      dropout_rate: dropout rate applied after activation and after output proj.
      swiglu:       if True, use SwiGLU; else standard GELU MLP.
      parity_2over3:
                    if True and swiglu=True, set hidden = (2/3)*mlp_ratio*d_model
                    to roughly match parameter count of a GELU MLP with mlp_ratio.
      dtype:        param/compute dtype.
    """
    d_model: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        """
        Args:
          x:            (..., d_model) input activations.
          deterministic:
                        True disables dropout (eval); False enables dropout (train).

        Returns:
          y:            (..., d_model) output activations, same shape as input.
        """
        # Choose hidden size
        mult = self.mlp_ratio
        if self.swiglu and self.parity_2over3:
            mult = self.mlp_ratio * (2.0 / 3.0)  # param parity with GELU MLP

        hidden = int(self.d_model * mult)

        if self.swiglu:
            # SwiGLU: Dense -> split -> u * silu(v)
            pre = nn.Dense(
                2 * hidden, dtype=self.dtype, name="fc_in"
            )(x)  # (..., 2H)
            u, v = jnp.split(pre, 2, axis=-1)     # (..., H), (..., H)
            h = u * jax.nn.silu(v)                # (..., H)
        else:
            # Standard GELU MLP
            h = nn.Dense(hidden, dtype=self.dtype, name="fc_in")(x)
            h = nn.gelu(h)

        h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        y = nn.Dense(self.d_model, dtype=self.dtype, name="fc_out")(h)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        return y

# ---------- axial attention layers ----------
class GroupedQueryAttention(nn.Module):
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    deterministic: bool = True
    qk_norm_type: str | None = None  # "qknorm", or "quest"
    is_causal: bool = False 
    rope_theta: float = 10000.0

    def setup(self):
        assert self.dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        self.to_q = nn.Dense(self.dim, use_bias=False)
        self.to_kv = nn.Dense(2 * kv_dim, use_bias=False)
        self.to_out = nn.Dense(self.dim, use_bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.qk_norm_type == 'qknorm':
            self.q_ln = nn.RMSNorm(use_scale=True)
            self.k_ln = nn.RMSNorm(use_scale=True)

        self.rope = RotaryEmbedding1D(
            dim=head_dim,
            theta=self.rope_theta
        )

    def __call__(self, x, mask, *args, cache: Optional[KVCache] = None):
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
            q = self.q_ln(q)
            k = self.k_ln(k)
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
        out = self.dropout(out, deterministic=self.deterministic)
        
        return out, new_cache

class SpaceSelfAttention(nn.Module):
    """Space self-attention with modality routing."""
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool, cache: Optional[KVCache] = None):
        # x: (B, T, S, D)  -> attention across S within each (B,T)
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B T) S D")

        # cache is not used for space attention (non-causal)
        out, _ = GroupedQueryAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            rope_theta=self.rope_theta,
            deterministic=deterministic,
            is_causal=False,
        )(x, mask=mask, cache=None)

        out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
        return out, None  # Return None for cache consistency

class TimeSelfAttention(nn.Module):
    """Time self-attention."""
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool, cache: Optional[KVCache] = None):
        # mask does nothing, but is required for API consistency
        # x: (B, T, S, D) -> attention across T, causal
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B S) T D")

        out, new_cache = GroupedQueryAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            rope_theta=self.rope_theta,
            deterministic=deterministic,
            is_causal=True,
        )(x, mask=None, cache=cache)

        out = rearrange(out, "(B S) T D -> B T S D", B=B, S=S)
        return out, new_cache

# ---------- a single block-causal layer ----------
class BlockCausalLayer(nn.Module):
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    mlp_ratio: float = 4.0
    layer_index: int = 0
    time_every: int = 4
    rope_theta: float = 10000.0

    def setup(self):
        self.norm = RMSNorm()

        # --- Time or space attention ---
        self.use_time = (self.layer_index + 1) % self.time_every == 0
        attention_module = TimeSelfAttention if self.use_time else SpaceSelfAttention
        self.attn = attention_module(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            rope_theta=self.rope_theta,
        )

        # --- MLP ---
        self.norm_mlp = RMSNorm()
        self.mlp = MLP(self.dim, self.mlp_ratio, self.dropout_rate)

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool, cache: Optional[KVCache] = None):
        # --- Attention (time or space, depending on layer_index) ---
        y = self.norm(x)
        y, new_cache = self.attn(y, mask=mask, deterministic=deterministic, cache=cache)
        x = x + y

        # --- MLP ---
        y = self.norm_mlp(x)
        y = self.mlp(y, deterministic=deterministic)
        x = x + y
        return x, new_cache

# ---------- the transformer stack ----------

class BlockCausalTransformer(nn.Module):
    d_model: int
    n_heads: int
    n_kv_heads: int
    depth: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    mlp_ratio: float = 4.0
    time_every: int = 4
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool, caches: Optional[Dict[int, KVCache]] = None):
        """
        Args:
            x: input tensor
            mask: attention mask
            deterministic: whether to use deterministic mode (no dropout)
            caches: optional dict mapping layer_index -> KVCache
        
        Returns:
            output tensor and updated caches (always returns tuple, caches can be None)
        """
        new_caches = {} if caches is not None else None
        
        for i in range(self.depth):
            time_index = i // self.time_every
            is_time_layer = (i + 1) % self.time_every == 0
            cache_i = caches.get(time_index) if caches is not None and is_time_layer else None
            x, new_cache_i = BlockCausalLayer(
                self.d_model, self.n_heads, self.n_kv_heads,
                dropout_rate=self.dropout_rate, qk_norm_type=self.qk_norm_type,
                mlp_ratio=self.mlp_ratio, layer_index=i, time_every=self.time_every,
                rope_theta=self.rope_theta,
            )(x, mask=mask, deterministic=deterministic, cache=cache_i)
            
            if new_caches is not None and new_cache_i is not None:
                new_caches[time_index] = new_cache_i
        
        return x, new_caches

class Encoder(nn.Module):
    d_model: int
    n_latents: int
    patch_size: int
    n_heads: int
    n_kv_heads: int
    depth: int
    d_bottleneck: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    mlp_ratio: float = 4.0
    time_every: int = 4
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    rope_theta: float = 10000.0

    def setup(self):
        self.patch_proj = nn.Dense(self.d_model, name="patch_proj")
        self.bottleneck_proj = nn.Dense(self.d_bottleneck, name="bottleneck_proj")

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            depth=self.depth,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            rope_theta=self.rope_theta,
        )
        self.mask_and_replace = MAEReplacer(name="mae", p_min=self.mae_p_min, p_max=self.mae_p_max)
        self.latents = self.param("latents_enc", nn.initializers.normal(0.02), (self.n_latents, self.d_model))

    @nn.compact
    def __call__(self, videos, *, deterministic: bool = True) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1) Make patches and project to D_model
        B, T, H, W, C = videos.shape
        patch_tokens = patchify(videos, patch=self.patch_size)
        proj_patches = self.patch_proj(patch_tokens)  # (B,T,Np,D)

        # 2) MAE mask-and-replace on patch tokens (encoder input only)
        proj_patches_masked, patch_mask, keep_prob = self.mask_and_replace(proj_patches)
        # patch_mask is (B,T,Np,1), need to expand to pixels (B,T,Np, P*P)
        patch_mask_expanded = jnp.repeat(patch_mask, self.patch_size**2, axis=-1)
        frame_mask = unpatchify(patch_mask_expanded, self.patch_size, H, W)

        # 3) Prepend learned latents (owned here)
        B, T = proj_patches_masked.shape[:2]
        latents = repeat(self.latents, "... -> b t ...", b=B, t=T)
        tokens = jnp.concatenate([latents, proj_patches_masked], axis=2)  # (B,T,S=(Np+Nl),D)

        # Flax MHA mask shape can be (batch, num_heads, q_len, k_len). We want one mask per (B*T).
        layout = TokenLayout((
            (Modality.LATENT, self.n_latents),
            (Modality.IMAGE, patch_tokens.shape[-2]),
            ))
        mask = layout.make_mask("encoder", B, T)

        # 5) Feed tokens into transformer

        encoded_tokens, _ = self.transformer(tokens, mask=mask, deterministic=deterministic)
        # print(f"encoded_tokens_btSd.shape: {encoded_tokens_btSd.shape}")


        # 6) Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, :self.n_latents]
        proj_tokens = nn.tanh(self.bottleneck_proj(latent_tokens))

        return proj_tokens, (frame_mask, keep_prob)  # keep mask if you want diagnostics

class Decoder(nn.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.

    Inputs:
      - z: (B, T, N_l, d_bottleneck)  -- encoder bottleneck output

    Config:
      - n_patches: number of patch query tokens to use in the decoder
      - d_patch:   dimensionality of each patch to reconstruct (D_patch)
      - d_model, n_heads, depth, dropout_rate, mlp_ratio, time_every, latents_only_time
        typically mirror the encoder.
    """
    d_model: int
    n_heads: int
    n_kv_heads: int
    depth: int
    n_latents: int
    patch_size: int
    d_patch: int
    H: int
    W: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    mlp_ratio: float = 4.0
    time_every: int = 4
    rope_theta: float = 10000.0

    def setup(self):
        self.n_patches = (self.H // self.patch_size) * (self.W // self.patch_size)
        self.up_proj = nn.Dense(self.d_model, name="up_proj")
        self.patch_head = nn.Dense(self.d_patch, name="patch_head") # (Np, D_patch)

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            depth=self.depth,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            rope_theta=self.rope_theta,
        )
        self.patch_queries = self.param("patch_queries", nn.initializers.normal(0.02),(self.n_patches, self.d_model)) # (Np, D)

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        B, T, N_l, d_bottleneck = z.shape

        # 1) Up-project latent bottleneck to d_model (per latent token)
        latents = self.up_proj(z)  # (B, T, N_l, D)

        # 2) Learned per-patch query tokens (owned by the decoder)
        patches = repeat(self.patch_queries, " ... -> b t ...", b=B, t=T)  # (B, T, Np, D)

        # 3) Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=-2)

        # 5) Make mask
        layout = TokenLayout((
            (Modality.LATENT, N_l),
            (Modality.IMAGE, self.n_patches)
            ))
        mask = layout.make_mask("decoder", B, T)

        x, _ = self.transformer(tokens, mask=mask, deterministic=deterministic)
        # 6) Prediction head over the patch-query slice

        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = self.patch_head(x_patches)  # (B, T, Np, D_patch)
        out_frames = unpatchify(pred_btnd, patch=self.patch_size, H=self.H, W=self.W)
        return out_frames

class Tokenizer(nn.Module):
    config: TokenizerConfig

    def setup(self):
        # Encoder configuration
        enc_kwargs = asdict(self.config.encoder)
        dec_kwargs = asdict(self.config.decoder)
        dec_kwargs["H"] = self.config.dataset.H
        dec_kwargs["W"] = self.config.dataset.W

        self.encoder = Encoder(**enc_kwargs)
        self.decoder = Decoder(**dec_kwargs)

    @nn.compact
    def __call__(self, videos, deterministic: bool = True):
        z, aux = self.encoder(videos, deterministic=deterministic)
        recon = self.decoder(z, deterministic=deterministic)
        return recon, aux

    def encode(self, videos, deterministic: bool = True):
        return self.encoder(videos, deterministic=deterministic)
        
    def decode(self, z, deterministic: bool = True):
        return self.decoder(z, deterministic=deterministic)

    @classmethod
    def from_pretrained(cls, path, *, load_for_training: bool = False):
        mngr = make_manager(path, item_names=("meta","state"))
        
        # 1. Restore metadata to get config
        # We need to find the latest step first
        latest = mngr.latest_step()
        if latest is None:
            raise ValueError("No checkpoint found")
            
        # We restore just meta first
        restored_meta = mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
        cfg_dict = restored_meta.meta["cfg"]
        cfg_dict = recursive_list_to_tuple(cfg_dict)
        cfg = from_dict(TokenizerConfig, cfg_dict)
        
        # 2. Init to get structure (params + constants)
        tokenizer = cls(cfg)
        rng = jax.random.PRNGKey(0)
        rng_params, rng_mae, rng_drop = jax.random.split(rng, 3)
        dummy = jax.random.uniform(rng, (1, cfg.dataset.T, cfg.dataset.H, cfg.dataset.W, cfg.dataset.C))
        variables = tokenizer.init(
            {"params": rng_params, "mae": rng_mae, "dropout": rng_drop},
            dummy,
            deterministic=True
        )
        
        # 3. Restore parameters using keys from initialized variables (so shapes/types match)
        # Note: We only restore 'params', keeping 'constants' from init
        restored = mngr.restore(latest, args=ocp.args.Composite(
            state=ocp.args.StandardRestore(None)
        ))
        
        loaded_params = restored.state
        
        # 4. Merge loaded params into variables
        variables = unfreeze(variables)
        variables["params"] = loaded_params["params"]
        variables = freeze(variables)

        return tokenizer, variables, cfg

class ActionEncoder(nn.Module):
    d_model: int
    n_keyboard: int = 5  # up, down, left, right, null (categorical actions)

    @nn.compact
    def __call__(
        self,
        actions: Optional[jnp.ndarray],           # (B, T) int32 in [0, n_keyboard)
        batch_time_shape: Optional[Tuple[int,int]] = None,
        as_tokens: bool = True,
    ):
        # Base "action token" embedding (used always)
        base_emb = self.param(
            'base_action_emb', nn.initializers.normal(0.02), (self.d_model,)
        )

        if actions is None:
            # unlabeled videos: just broadcast base embedding
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = jnp.broadcast_to(base_emb, (B, T, self.d_model))
        else:
            # embed categorical actions
            emb_key = nn.Embed(self.n_keyboard, self.d_model, name="emb_key")(actions)
            out = emb_key + base_emb  # broadcast add

        if as_tokens:
            # expand a token axis (S_a = 1)
            out = out[:, :, None, :]

        return out

class Dynamics(nn.Module):
    config: DynamicsModelConfig

    def setup(self):
        # Extract params from configs
        self.d_model = self.config.d_model
        self.d_bottleneck = self.config.d_bottleneck
        self.depth = self.config.depth
        self.n_heads = self.config.n_heads
        self.n_kv_heads = self.config.n_kv_heads
        packing_factor = self.config.packing_factor
        self.n_register = self.config.n_register
        self.n_agent = self.config.n_agent
        self.k_max = self.config.k_max
        
        # Defaults
        dropout_rate = self.config.dropout_rate
        qk_norm_type = self.config.qk_norm_type
        mlp_ratio = self.config.mlp_ratio
        time_every = self.config.time_every
        rope_theta = self.config.rope_theta

        # Want to transform bottleneck inputs (B, T, N_b, D_b) to (B, T, N_b/packing_factor, D_b*packing_factor)
        self.spatial_proj = nn.Dense(self.d_model, name="proj_spatial") # converts spatial tokens, of dim d_spatial to d_model
        
        self.register_tokens = self.param(
            "register_tokens",
            nn.initializers.normal(0.02),
            (self.n_register, self.d_model),
        )
        self.action_encoder = ActionEncoder(d_model=self.d_model)


        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            depth=self.depth,
            dropout_rate=dropout_rate,
            qk_norm_type=qk_norm_type,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            rope_theta=rope_theta,
        )

        # -------- Discrete embeddings for shortcut conditioning --------
        # Step size d ∈ {1, 1/2, 1/4, ..., 1/256}
        # We index steps by: step_idx = log2(1/d) ∈ {0, 1, 2, ...,7, 8}
        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embed(self.num_step_bins, self.d_model, name="step_embed")

        # Signal level τ ∈ {0, 1/d, 2/d, ..., 1 - 1/d} (grid length = 1/d)
        # We use a *shared* table with  bins and only use the first (1/d) entries for a given d.
        self.signal_embed = nn.Embed(self.k_max, self.d_model, name="signal_embed")
        self.flow_x_head = nn.Dense(self.d_bottleneck * packing_factor, name="flow_x_head", kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros)  # zero-init

    def create_static_caches(self, 
                             batch_size: int, 
                             window_size: int = 1024,
                             dtype=jnp.float32) -> Dict[int, KVCache]:
        """
        Creates concrete, zero-filled buffers for JIT compilation.

        Args:
            batch_size: Global batch size (B)
            window_size: Maximum sequence length (T) to support.
        """
        # Time attention sees (B*S) independent sequences. S includes all tokens (action, signal, step, spatial, register, agent)
        S_total = 1 + 1 + 1 + self.n_spatial + self.n_register + (self.n_agent if self.n_agent > 0 else 0)

        flattened_batch_size = batch_size * S_total
        head_dim = self.d_model // self.n_heads

        caches = {}
        for i in range(self.depth):
            time_index, time_offset = divmod(i, self.time_every)
            if time_offset == 0:
                caches[time_index] = KVCache.init(
                    batch_size=flattened_batch_size,
                    window_size=window_size,
                    num_kv_heads=self.n_kv_heads,
                    head_dim=head_dim,
                    dtype=dtype
                )
        return caches

    @nn.compact
    def __call__(
        self,
        actions,             # (B,T)
        step_idxs,           # (B,T)
        signal_idxs,         # (B,T)
        packed_enc_tokens,   # (B,T,n_s,d_spatial)
        *,
        agent_tokens: Optional[jnp.ndarray] = None,  # (B,T,n_agent,D) or None
        deterministic: bool = True,
        caches: Optional[Dict[int, KVCache]] = None,
    ):
        """
        Args:
          packed_enc_tokens:      (B, T, n_spatial, d_spatial) packed encoder tokens
          actions:    (B, T) int32 in [0, n_keyboard) raw action tokens
          steps:      (B, T) float32 — step sizes, 1/2^x
          signals:    (B, T) float32 - signal values, grid that is reachable by current step size
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
            self.register_tokens[None, None, ...],  # (1,1,n_register,d_model)
            (B, T, self.n_register, self.d_model),
        )

        # --- 4) Shortcut embeddings (discrete lookup)
        step_tok   = self.step_embed(step_idxs)[:, :, None, :]         # (B, T, 1, d_model)
        signal_tok = self.signal_embed(signal_idxs)[:, :, None, :]     # (B, T, 1, d_model)
        
        # --- 5) Concatenate in your declared layout order
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = jnp.zeros((B, T, self.n_agent, self.d_model), dtype=spatial_tokens.dtype)
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, agent_tokens]
        else:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens]
        tokens = jnp.concatenate(toks, axis=2)                    # (B,T,S,D)
        
        # make the layout for masking
        modalities = [Modality.ACTION, Modality.SHORTCUT_SIGNAL, Modality.SHORTCUT_STEP, Modality.SPATIAL, Modality.REGISTER, Modality.AGENT]
        segments = tuple((modality, tok.shape[2]) for modality, tok in zip(modalities, toks))
        layout = TokenLayout(segments)
        mask = layout.make_mask("wm_agent", B, T)

        x, new_caches = self.transformer(tokens, mask, deterministic=deterministic, caches=caches)

        spatial_tokens = x[:, :, layout.slices()[Modality.SPATIAL], :]
        x1_hat = self.flow_x_head(spatial_tokens)
        h_t = x[:, :, layout.slices()[Modality.AGENT], :] if self.n_agent > 0 else None  # (B,T,n_agent,D) or None
        return x1_hat, h_t, new_caches

class TaskEmbedder(nn.Module):
    d_model: int
    n_agent: int = 1
    use_ids: bool = True     # True: task is int ids; False: task is vector
    n_tasks: int = 128       # only used if use_ids=True
    d_task: int = 64         # only used if use_ids=False

    @nn.compact
    def __call__(self, task, B: int, T: int):
        """
        If use_ids=True:
            task: (B,) int32 ids in [0, n_tasks)
        Else:
            task: (B, d_task) float32 vector

        Returns agent tokens: (B, T, n_agent, d_model)
        """
        if self.use_ids:
            emb = nn.Embed(self.n_tasks, self.d_model, name="task_table")(task)  # (B, D)
        else:
            emb = nn.Dense(self.d_model, name="task_proj")(task)                 # (B, D)

        # Learned base + optional small MLP to decouple from raw table
        base = self.param("agent_base", nn.initializers.normal(0.02), (self.d_model,))
        x = emb + base[None, :]

        # Replicate across time and agent slots
        x = jnp.broadcast_to(x[:, None, None, :], (B, T, self.n_agent, self.d_model))
        return x

# === Phase B heads (use existing MLP) =========================================

class PolicyHeadMTP(nn.Module):
    """Multi-Token action prediction.
    Input:  h_t (B, T, D)  -- agent readouts (pool n_agent first if needed)
    Output: logits (B, T, L, A)
    """
    d_model: int
    action_dim: int
    L: int = 8
    kind: str = "categorical"  # or "vbinary"
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        # Feature projector (D -> D) using your MLP
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            swiglu=self.swiglu,
            parity_2over3=self.parity_2over3,
            dtype=self.dtype,
        )
        # Single matmul that produces all L offsets at once: (… , D) -> (…, L, A)
        self.out = nn.DenseGeneral(
            features=(self.L, self.action_dim),
            axis=-1,
            dtype=self.dtype,
            name="out",
        )

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.projector(h_t, deterministic=deterministic)  # (B, T, D)
        logits = self.out(x)                                  # (B, T, L, A)
        return logits  # softmax/sigmoid applied at loss-time based on `kind`


class RewardHeadMTP(nn.Module):
    """Multi-Token reward prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, L, K), centers (K,)
    """
    d_model: int
    L: int = 8
    num_bins: int = 101
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32
    # log-space bounds for symexp bins (tune per dataset)
    log_low: float = -8.0
    log_high: float = 8.0

    def setup(self):
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            swiglu=self.swiglu,
            parity_2over3=self.parity_2over3,
            dtype=self.dtype,
        )
        self.out = nn.DenseGeneral(
            features=(self.L, self.num_bins),
            axis=-1,
            dtype=self.dtype,
            name="out",
        )
        # Precompute bin centers as a constant (share across calls/checkpoints)
        # Simple choice: uniform in log-space, then exponentiate symmetrically.
        log_edges = jnp.linspace(self.log_low, self.log_high, self.num_bins)
        # centers ~ same length for convenience (pad to K if using edges-midpoints):
        centers = log_edges
        self.centers_var = self.variable("constants", "symexp_centers_log", lambda: centers)

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t, deterministic=deterministic)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, L, K)
        centers_log = self.centers_var.value                   # (K,)
        return logits, centers_log


class ValueHead(nn.Module):
    """Value prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, K), centers (K,)
    """
    d_model: int
    num_bins: int = 101
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32
    # log-space bounds for symexp bins (tune per dataset)
    log_low: float = -8.0
    log_high: float = 8.0

    def setup(self):
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            swiglu=self.swiglu,
            parity_2over3=self.parity_2over3,
            dtype=self.dtype,
        )
        self.out = nn.DenseGeneral(
            features=self.num_bins,
            axis=-1,
            dtype=self.dtype,
            name="out",
        )
        # Precompute bin centers as a constant (share across calls/checkpoints)
        # Simple choice: uniform in log-space, then exponentiate symmetrically.
        log_edges = jnp.linspace(self.log_low, self.log_high, self.num_bins)
        # centers ~ same length for convenience (pad to K if using edges-midpoints):
        centers = log_edges
        self.centers_var = self.variable("constants", "symexp_centers_log", lambda: centers)

    @nn.compact
    def __call__(self, h_t: jnp.ndarray, *, deterministic: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t, deterministic=deterministic)   # (B, T, D)
        logits = self.out(x)                                   # (B, T, K)
        centers_log = self.centers_var.value                   # (K,)
        return logits, centers_log
