from functools import lru_cache
from dataclasses import asdict
import jax.numpy as jnp
import flax.linen as nn
import jax
import time
from flax.core import FrozenDict
import flax
from enum import IntEnum
from typing import Optional, Tuple, Any
from einops import rearrange, repeat
import math
from .utils import Modality, TokenLayout
from .data import patchify, unpatchify
from .configs import TokenizerConfig



    



@lru_cache(maxsize=8)
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    Returns:
        freqs_cos: (end, dim//2)
        freqs_sin: (end, dim//2)
    """
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)  # (end, dim//2)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rotary_emb(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cos: jnp.ndarray, freqs_sin: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply Rotary Positional Embeddings (RoPE) to queries and keys using real sin/cos.
    xq: (B, T, N, H)
    xk: (B, S, K, H)
    freqs_cos: (L, H/2)
    freqs_sin: (L, H/2)
    """
    # Rearrange to (..., H/2, 2)
    xq_pairs = rearrange(xq, '... (d two) -> ... d two', two=2)
    xk_pairs = rearrange(xk, '... (d two) -> ... d two', two=2)

    xq_r, xq_i = xq_pairs[..., 0], xq_pairs[..., 1]
    xk_r, xk_i = xk_pairs[..., 0], xk_pairs[..., 1]

    T, S = xq.shape[1], xk.shape[1]
    
    # Broadcast freqs (L, H/2) -> (1, L, 1, H/2)
    cos_q = freqs_cos[None, :T, None, :]
    sin_q = freqs_sin[None, :T, None, :]
    cos_k = freqs_cos[None, :S, None, :]
    sin_k = freqs_sin[None, :S, None, :]

    # Rotation:
    # x' = x cos - y sin
    # y' = x sin + y cos
    xq_out_r = xq_r * cos_q - xq_i * sin_q
    xq_out_i = xq_r * sin_q + xq_i * cos_q
    
    xk_out_r = xk_r * cos_k - xk_i * sin_k
    xk_out_i = xk_r * sin_k + xk_i * cos_k

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
            self.q_ln = nn.LayerNorm(use_bias=True, use_scale=True)
            self.k_ln = nn.LayerNorm(use_bias=True, use_scale=True)

    def __call__(self, x, mask, *args):
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
        kv = self.to_kv(x)
        q = rearrange(q, "B T (N H) -> B T N H", N=self.num_heads)
        k, v = rearrange(kv, "B S (C K H) -> C B S K H", C=2, K=self.num_kv_heads)

        head_dim = q.shape[-1]
        max_len = max(q.shape[1], k.shape[1])
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_len, self.rope_theta, dtype=q.dtype)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        scale = q.shape[-1] ** -0.5
        if self.qk_norm_type == 'qknorm':
            q = self.q_ln(q)
            k = self.k_ln(k)
        elif self.qk_norm_type == 'quest':
            # claims to beat qknorm https://openreview.net/pdf?id=HkztQWZfl2
            k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
            scale = 1.0 

        attn = jax.nn.dot_product_attention(q, k, v, mask=mask, scale=scale, is_causal=self.is_causal)  # TODO: try setting implementation="cudnn"
        attn = rearrange(attn, "B T N H -> B T (N H)")

        out = self.to_out(attn)
        return self.dropout(out, deterministic=self.deterministic)

class SpaceSelfAttentionModality(nn.Module):
    """Space self-attention with modality routing."""
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool):
        # x: (B, T, S, D)  -> attention across S within each (B,T)
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B T) S D")

        out = GroupedQueryAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            rope_theta=self.rope_theta,
            deterministic=deterministic,
            is_causal=False,
        )(x, mask=mask)

        out = rearrange(out, "(B T) S D -> B T S D", B=B, T=T)
        return out

class TimeSelfAttention(nn.Module):
    """Time self-attention."""
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool):
        # mask does nothing, but is required for API consistency
        # x: (B, T, S, D) -> attend across T, causal
        B, T, S, D = x.shape
        x = rearrange(x, "B T S D -> (B S) T D")
        out = GroupedQueryAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout_rate=self.dropout_rate,
            qk_norm_type=self.qk_norm_type,
            rope_theta=self.rope_theta,
            deterministic=deterministic,
            is_causal=True,
        )(x, mask=None)
        out = rearrange(out, "(B S) T D -> B T S D", B=B, S=S)
        return out

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
        attention_module = TimeSelfAttention if self.use_time else SpaceSelfAttentionModality
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
    def __call__(self, x, mask, *, deterministic: bool):
        # --- Space attention (within timestep, modality-aware) ---
        y = self.norm(x)
        y = self.attn(y, mask=mask, deterministic=deterministic)
        x = x + y

        # --- MLP ---
        y = self.norm_mlp(x)
        y = self.mlp(y, deterministic=deterministic)
        x = x + y
        return x
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
    def __call__(self, x, mask, *, deterministic: bool):
        for i in range(self.depth):
            x = BlockCausalLayer(
                self.d_model, self.n_heads, self.n_kv_heads,
                dropout_rate=self.dropout_rate, qk_norm_type=self.qk_norm_type,
                mlp_ratio=self.mlp_ratio, layer_index=i, time_every=self.time_every,
                rope_theta=self.rope_theta,
            )(x, mask=mask, deterministic=deterministic)
        return x

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
        encoded_tokens = self.transformer(tokens, mask=mask, deterministic=deterministic)

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

        # 6) Axial block-causal transformer
        x = self.transformer(tokens, mask=mask, deterministic=deterministic)
        # 7) Prediction head over the patch-query slice
        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = nn.tanh(self.patch_head(x_patches))  # (B,T,Np,D_patch)
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
    d_model: int              # dimensionality of each token
    d_bottleneck: int         # dimensionality of the input bottleneck space
    d_spatial: int            # dimensionality of each spatial token input
    n_spatial: int            # number of spatial tokens
    n_register: int           # number of learned register tokens
    n_agent: int              # number of agent tokens
    n_heads: int
    n_kv_heads: int
    depth: int
    k_max: int                 # maximum number of sampling steps (defines finest step 1/)
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    mlp_ratio: float = 4.0
    time_every: int = 4
    rope_theta: float = 10000.0

    def setup(self):
        # Want to transform bottleneck inputs (B, T, N_b, D_b) to (B, T, N_b/packing_factor, D_b*packing_factor)
        assert self.d_spatial % self.d_bottleneck == 0
        self.spatial_proj = nn.Dense(self.d_model, name="proj_spatial") # converts spatial tokens, of dim d_spatial to d_model
        self.register_tokens = self.param(
            "register_tokens",
            nn.initializers.normal(0.02),
            (self.n_register, self.d_model),
        )
        self.action_encoder = ActionEncoder(d_model=self.d_model)

        # Two separate tokens for shortcut conditioning (your current layout):
        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),   # τ (signal level) token
            (Modality.SHORTCUT_STEP, 1),     # d (step size) token
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))
        self.layout = TokenLayout(segments=tuple(segments))
        self.spatial_slice = self.layout.slices()[Modality.SPATIAL]
        self.agent_slice  = self.layout.slices().get(Modality.AGENT, slice(0,0))  # safe if n_agent==0
        self.modality_ids = self.layout.modality_ids()

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

        # -------- Discrete embeddings for shortcut conditioning --------
        # Step size d ∈ {1, 1/2, 1/4, ..., 1/256}
        # We index steps by: step_idx = log2(1/d) ∈ {0, 1, 2, ...,7, 8}
        self.num_step_bins = int(math.log2(self.k_max)) + 1
        self.step_embed = nn.Embed(self.num_step_bins, self.d_model, name="step_embed")

        # Signal level τ ∈ {0, 1/d, 2/d, ..., 1 - 1/d} (grid length = 1/d)
        # We use a *shared* table with  bins and only use the first (1/d) entries for a given d.
        self.signal_embed = nn.Embed(self.k_max + 1, self.d_model, name="signal_embed")
        self.flow_x_head = nn.Dense(self.d_spatial, name="flow_x_head", kernel_init=nn.initializers.zeros,
                            bias_init=nn.initializers.zeros)  # zero-init

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
    ):
        """
        Args:
          packed_enc_tokens:      (B, T, n_spatial, d_spatial) packed encoder tokens
          actions:    (B, T) int32 in [0, n_keyboard) raw action tokens
          steps:      (B, T) float32 — step sizes, 1/2^x
          signals:    (B, T) float32 - signal values, grid that is reachable by current step size

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
        step_tok   = self.step_embed(step_idxs)[:, :, None, :]      # (B, T, 1, d_model)
        signal_tok = self.signal_embed(signal_idxs)[:, :, None, :]     # (B, T, 1, d_model)
        
        # --- 5) Concatenate in your declared layout order
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = jnp.zeros((B, T, self.n_agent, self.d_model), dtype=spatial_tokens.dtype)
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, agent_tokens]
        else:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens]
        tokens = jnp.concatenate(toks, axis=2)                    # (B,T,S,D)


        
        mask = self.layout.make_mask("wm_agent", B, T)
        x = self.transformer(tokens, mask, deterministic=deterministic)
        spatial_tokens = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_tokens)
        h_t = x[:, :, self.agent_slice, :] if self.n_agent > 0 else None  # (B,T,n_agent,D) or None
        return x1_hat, h_t

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
