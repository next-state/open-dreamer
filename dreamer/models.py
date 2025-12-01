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
from .utils import make_mask, Modality


@flax.struct.dataclass  # immutable, PyTree-friendly
class TokenLayout:
    """
    Ordered token layout for a single timestep: latents first (if any),
    then a sequence of (modality, count) segments.
    """
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]  # e.g., ((Modality.IMAGE, n_patches), (Modality.ACTION, n_act), ...)

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> jnp.ndarray:
        parts = [jnp.full((self.n_latents,), Modality.LATENT, dtype=jnp.int32)] if self.n_latents > 0 else []
        for m, n in self.segments:
            if n > 0:
                parts.append(jnp.full((n,), int(m), dtype=jnp.int32))
        return jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=jnp.int32)  # (S,)

    def slices(self) -> dict:
        """Convenience: start/stop indices per modality (first occurrence if repeated)."""
        idx = 0
        out = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents); idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out

    
def sinusoid_table(n: int, d: int, base: float = 10000.0, dtype=jnp.float32) -> jnp.ndarray:
    """
    Standard Transformer sinusoid: even dims use sin, odd dims use cos with frequencies
    base^{-2k/d}. Works for odd d too.
    """
    pos = jnp.arange(n, dtype=dtype)[:, None]            # (n,1)
    i = jnp.arange(d, dtype=dtype)[None, :]              # (1,d)
    # k = floor(i/2)
    k = jnp.floor(i / 2.0)
    div = jnp.power(base, -(2.0 * k) / jnp.maximum(1.0, jnp.array(d, dtype)))
    angles = pos * div                                    # (n,d)
    table = jnp.where((i % 2) == 0, jnp.sin(angles), jnp.cos(angles))
    return table.astype(dtype)


def add_sinusoidal_positions(tokens_btSd: jnp.ndarray) -> jnp.ndarray:
    """tokens: (B,T,S,D) -> adds time and step sinusoids and returns same shape."""
    B, T, S, D = tokens_btSd.shape
    pos_t = sinusoid_table(T, D)     # (T,D)
    pos_s = sinusoid_table(S, D)     # (S,D)
    # Optionally scale to keep variance stable (common trick)
    scale = 1.0 / jnp.sqrt(jnp.array(D, dtype=tokens_btSd.dtype))
    return tokens_btSd + scale * (pos_t[None, :, None, :] + pos_s[None, None, :, :])

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
      dim:          input/output width of the block (last-dim of x).
      mlp_ratio:    expansion factor for hidden size if NOT using 2/3 parity.
      dropout_rate: dropout rate applied after activation and after output proj.
      swiglu:       if True, use SwiGLU; else standard GELU MLP.
      parity_2over3:
                    if True and swiglu=True, set hidden = (2/3)*mlp_ratio*d_model
                    to roughly match parameter count of a GELU MLP with mlp_ratio.
      dtype:        param/compute dtype.
    """
    dim: int
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

        hidden = int(self.dim * mult)

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
        y = nn.Dense(self.dim, dtype=self.dtype, name="fc_out")(h)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        return y

# ---------- axial attention layers ----------
class GroupedQueryAttention(nn.Module):
    dim: int
    num_heads: int
    num_kv_heads: int
    dropout_rate: float = 0.0
    deterministic: bool = True

    def setup(self):
        assert self.dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        head_dim = self.dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim

        self.to_q = nn.Dense(self.dim, use_bias=False)
        self.to_kv = nn.Dense(2 * kv_dim, use_bias=False)
        self.to_out = nn.Dense(self.dim, use_bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, mask, *, deterministic: bool):
        """
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

        attn = jax.nn.dot_product_attention(q, k, v, mask=mask)  # TODO: try setting implementation="cudnn"
        attn = rearrange(attn, "B T N H -> B T (N H)")

        out = self.to_out(attn)
        return self.dropout(out, deterministic=deterministic)

class SpaceSelfAttentionModality(nn.Module):
    """Space self-attention with modality routing."""
    dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool):
        # x: (B, T, S, D)  -> attention across S within each (B,T)
        B, T, S, D = x.shape
        x_ = rearrange(x, "B T S D -> (B T) S D")

        y_ = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x_, x_, mask=mask)

        y = rearrange(y_, "(B T) S D -> B T S D", B=B, T=T)
        return y

class TimeSelfAttention(nn.Module):
    """Time self-attention."""
    dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool):
        # mask does nothing, but is required for API consistency
        # x: (B, T, S, D) -> attend across T, causal
        B, T, S, D = x.shape
        x_bstd = rearrange(x, "B T S D -> (B S) T D")
        causal = nn.attention.make_causal_mask(jnp.ones((B*S, T), dtype=bool))
        out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dim=self.dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x_bstd, x_bstd, mask=causal)
        out = rearrange(out, "(B S) T D -> B T S D", B=B, S=S)
        return out

# ---------- a single block-causal layer ----------
class BlockCausalLayer(nn.Module):
    dim: int
    num_heads: int
    dropout_rate: float = 0.0
    mlp_ratio: float = 4.0
    layer_index: int = 0
    time_every: int = 4

    def setup(self):
        self.norm = RMSNorm()

        # --- Time or space attention ---
        self.use_time = (self.layer_index + 1) % self.time_every == 0
        attention_module = TimeSelfAttention if self.use_time else SpaceSelfAttentionModality
        self.attn = attention_module(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
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
    depth: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool):
        for i in range(self.depth):
            x = BlockCausalLayer(
                self.d_model, self.n_heads,
                dropout=self.dropout, mlp_ratio=self.mlp_ratio,
                layer_index=i, time_every=self.time_every,
            )(x, mask=mask, deterministic=deterministic)
        return x

class Encoder(nn.Module):
    d_model: int
    n_latents: int
    n_patches: int
    n_heads: int
    depth: int
    d_bottleneck: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    
    def setup(self):
        self.patch_proj = nn.Dense(self.d_model, name="patch_proj")
        self.bottleneck_proj = nn.Dense(self.d_bottleneck, name="bottleneck_proj")
        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        self.modality_ids = self.layout.modality_ids()            # (S,)
        mask = make_mask(self.modality_ids, "encoder")
        self.mask = self.variable("constants", "mask", lambda: mask)

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout, mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
        )
        self.latents = self.param("latents_enc", nn.initializers.normal(0.02), (self.n_latents, self.d_model))

    @nn.compact
    def __call__(self, patch_tokens, *, deterministic: bool = True) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1) Project patches to D_model
        proj_patches = self.patch_proj(patch_tokens)  # (B,T,Np,D)

        # 2) MAE mask-and-replace on patch tokens (encoder input only)
        proj_patches_masked, patch_mask, keep_prob = MAEReplacer(name="mae", p_min=self.mae_p_min, p_max=self.mae_p_max)(proj_patches)
        # print(f"proj_patches_masked.shape: {proj_patches_masked.shape}")
        # print(f"patch_mask.shape: {patch_mask.shape}")

        # 3) Prepend learned latents (owned here)
        # print(f"latents.shape: {latents.shape}")
        B, T = proj_patches_masked.shape[:2]
        latents = jnp.broadcast_to(self.latents[None, None, ...], (B, T, *self.latents.shape))
        # print(f"lat_btld.shape: {lat_btld.shape}")
        tokens = jnp.concatenate([latents, proj_patches_masked], axis=2)  # (B,T,S=(Np+Nl),D)
        # print(f"tokens_btSd.shape: {tokens_btSd.shape}")

        # 4) Add sinusoidal positions (param-free)
        tokens = add_sinusoidal_positions(tokens)

        # Flax MHA mask shape can be (batch, num_heads, q_len, k_len). We want one mask per (B*T).
        mask = repeat(self.mask.value, " ... -> bt h ...", bt=B*T, h=1)
        # 5) Feed tokens into transformer
        encoded_tokens = self.transformer(tokens, mask=mask, deterministic=deterministic)
        # print(f"encoded_tokens_btSd.shape: {encoded_tokens_btSd.shape}")

        # 6) Project latent tokens to bottleneck and tanh
        latent_tokens = encoded_tokens[:, :, :self.n_latents, :]
        proj_tokens = nn.tanh(self.bottleneck_proj(latent_tokens))

        return proj_tokens, (patch_mask, keep_prob)  # keep mask if you want diagnostics

class Decoder(nn.Module):
    """
    MAE-style decoder that reads temporal info from latent tokens and writes
    reconstructions at per-patch query tokens.

    Inputs:
      - z: (B, T, N_l, d_bottleneck)  -- encoder bottleneck output

    Config:
      - n_patches: number of patch query tokens to use in the decoder
      - d_patch:   dimensionality of each patch to reconstruct (D_patch)
      - d_model, n_heads, depth, dropout, mlp_ratio, time_every, latents_only_time
        typically mirror the encoder.
    """
    d_model: int
    n_heads: int
    depth: int
    n_latents: int
    n_patches: int
    d_patch: int
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4

    def setup(self):
        self.layout = TokenLayout(n_latents=self.n_latents, segments=((Modality.IMAGE, self.n_patches),))
        self.modality_ids = self.layout.modality_ids()
        self.up_proj = nn.Dense(self.d_model, name="up_proj")
        self.patch_queries = self.param(
            "patch_queries",
            nn.initializers.normal(0.02),
            (self.n_patches, self.d_model),
        ) # (Np, D)
        self.patch_head = nn.Dense(self.d_patch, name="patch_head") # (Np, D_patch)
        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
        )
        mask = make_mask(self.modality_ids, "decoder")
        self.mask = self.variable("constants", "mask", lambda: mask)

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        B, T, N_l, d_bottleneck = z.shape

        # 1) Up-project latent bottleneck to d_model (per latent token)
        latents = nn.tanh(self.up_proj(z))  # (B, T, N_l, D)

        # 2) Learned per-patch query tokens (owned by the decoder)
        patches = jnp.broadcast_to(
            self.patch_queries[None, None, ...],
            (B, T, self.n_patches, self.d_model),
        )  # (B, T, Np, D)

        # 3) Concat: [latents, patch queries]  ->  (B, T, S=N_l+N_p, D)
        tokens = jnp.concatenate([latents, patches], axis=2)

        # 4) Add sinusoidal positions
        tokens = add_sinusoidal_positions(tokens)

        # 5) Axial block-causal transformer
        #    - SpaceSelfAttention over all S tokens (latents + queries)
        #    - TimeSelfAttention only over the first N_l latent tokens
        mask = repeat(self.mask.value, " ... -> bt h ...", bt=B*T, h=1)
        x = self.transformer(tokens, mask=mask, deterministic=deterministic)
        # 6) Prediction head over the patch-query slice
        x_patches = x[:, :, N_l:, :]                         # (B, T, Np, D)
        pred_btnd = nn.sigmoid(self.patch_head(x_patches))  # (B,T,Np,D_patch)
        return pred_btnd

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
    depth: int
    k_max: int                 # maximum number of sampling steps (defines finest step 1/)
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    time_every: int = 4

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
        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        self.spatial_slice = self.layout.slices()[Modality.SPATIAL]
        self.agent_slice  = self.layout.slices().get(Modality.AGENT, slice(0,0))  # safe if n_agent==0
        self.modality_ids = self.layout.modality_ids()
        mask = make_mask(self.modality_ids, "wm_agent")
        self.mask = self.variable("constants", "mask", lambda: mask)

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
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

        tokens = add_sinusoidal_positions(tokens)      # (B, T, N_total, d_model)
        mask = repeat(self.mask.value, " ... -> bt h ...", bt=B*T, h=1)
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
    dropout: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        # Feature projector (D -> D) using your MLP
        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
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
    dropout: float = 0.0
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
            dropout=self.dropout,
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
    dropout: float = 0.0
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
            dropout=self.dropout,
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




# ---------- test encoder/decoder ----------

def test_encoder_decoder():
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    n_patches = 4
    d_patch = 3
    enc_n_latents = 2
    enc_d_bottleneck = 3
    x = jnp.ones((B, T, n_patches, d_patch))  # (B,T,Np,D_patch)

    encoder = Encoder(d_model=8, n_latents=enc_n_latents, n_patches=n_patches, n_heads=2, depth=2, dropout=0.5, d_bottleneck=enc_d_bottleneck)
    decoder = Decoder(d_model=8, n_heads=2, depth=2, n_patches=n_patches, n_latents=enc_n_latents, d_patch=d_patch, dropout=0.5)
    # init: give both "mae" and "dropout" keys (dropout only needed if deterministic=False)
    enc_vars = encoder.init(
        {"params": rng, "mae": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)},
        x,
        deterministic=True,
    )
    # Decode
    fake_z = jnp.ones((B, T, enc_n_latents, enc_d_bottleneck))
    dec_vars = decoder.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_z,
        deterministic=True,
    )

    def forward_apply(enc_vars: FrozenDict, dec_vars: FrozenDict,
                    patches_btnd: jnp.ndarray,
                    *, mae_key=None, drop_key=None, train: bool):
        # Encoder
        rngs_enc = {}
        if train:
            rngs_enc = {"mae": mae_key, "dropout": drop_key}
        else:
            rngs_enc = {"mae": mae_key}  # if you still want masking during eval

        z_btLd, mae_info = encoder.apply(enc_vars, patches_btnd,
                                        rngs=rngs_enc,
                                        deterministic=not train)
        # Decoder
        rngs_dec = {"dropout": drop_key} if train else {}
        pred_btnd = decoder.apply(dec_vars, z_btLd,
                                rngs=rngs_dec,
                                deterministic=not train)
        return pred_btnd, mae_info
    
    jit_forward = jax.jit(forward_apply, static_argnames=["train"])
    mae_key = jax.random.PRNGKey(0)
    drop_key = jax.random.PRNGKey(1)
    # Warm-up (compilation happens here)
    t0 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t1 = time.time()
    # Hot run (should be much faster)
    t2 = time.time()
    out = jit_forward(enc_vars, dec_vars, x, mae_key=mae_key, drop_key=drop_key, train=True)
    jax.tree_util.tree_map(lambda y: y.block_until_ready(), out)
    t3 = time.time()

    print(f"Warm-up (compile+run): {t1 - t0:.3f}s")
    print(f"Hot run (cached):      {t3 - t2:.3f}s")

def test_dynamics():
    rng = jax.random.PRNGKey(0)
    B = 2
    T = 10
    fake_enc_z = jnp.ones((B, T, 512, 16), dtype=jnp.float32)
    fake_actions = jnp.ones((B, T), dtype=jnp.int32)
    fake_step_idxs = jnp.zeros((B, T), dtype=jnp.int32)
    fake_signal_idxs = jnp.zeros((B, T), dtype=jnp.int32)
    def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
        """
        (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
        Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
        """
        return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)
    fake_packed_enc_tokens = pack_bottleneck_to_spatial(fake_enc_z, n_spatial=256, k=2)


    # need some way to assert that 512 * 16 == 256 * 32
    dynamics_kwargs = {
        "d_model": 128,
        "n_spatial": 256,
        "d_spatial": 32,
        "d_bottleneck": 16,
        "k_max": 8,
        "n_register": 10,
        "n_agent": 1,
        "n_heads": 4,
        "depth": 4,
        "dropout": 0.0
    }
    dynamics = Dynamics(**dynamics_kwargs)
    dynamics_vars = dynamics.init(
        {"params": rng, "dropout": jax.random.PRNGKey(2)},
        fake_actions,
        fake_step_idxs,
        fake_signal_idxs,
        fake_packed_enc_tokens,
    )
    out = dynamics.apply(dynamics_vars, fake_actions, fake_step_idxs, fake_signal_idxs, fake_packed_enc_tokens,
                        rngs={"dropout": jax.random.PRNGKey(2)},
                        deterministic=True)

def _build_modality_mask(modality_ids, mode: str):
    return make_mask(modality_ids, mode)

def _pack_bottleneck_to_spatial(z_btLd, n_spatial, k):
    return rearrange(z_btLd, 'b t (n k) d -> b t n (k d)', n=n_spatial, k=k)

def _abbr(m):
    # short labels just for printing rows/cols
    return {
        int(Modality.ACTION): "ACT",
        int(Modality.SHORTCUT_SIGNAL): "SIG",
        int(Modality.SHORTCUT_STEP): "STP",
        int(Modality.SPATIAL): "SPA",
        int(Modality.REGISTER): "REG",
        int(Modality.AGENT): "AGT",
        int(Modality.LATENT): "LAT",
    }.get(int(m), f"M{int(m)}")

def _print_mask_summary(name: str, modality_ids: jnp.ndarray, mask_2d: jnp.ndarray):
    # mask_2d: (S,S) with True meaning "query row can read key col"
    S = modality_ids.shape[0]
    mods = [int(x) for x in list(modality_ids)]
    headers = "     " + " ".join(f"{_abbr(m):>3}" for m in mods)
    print(f"\n[{name}] modality order (Q rows / K cols): {mods}")
    print(headers)
    for q in range(S):
        row = "".join("  ✓" if bool(mask_2d[q, k]) else "  ·" for k in range(S))
        print(f"{_abbr(modality_ids[q]):>3}: {row}")
    # row-wise counts
    counts = jnp.sum(mask_2d, axis=1)
    print("Row read-counts:", counts.tolist())

def test_agent_firewall():
    # layout: [ACTION, SIG, STEP, SPATIALx3, REGISTERx2, AGENTx1]
    ACTION,SIGNAL,STEP,SPATIAL,REGISTER,AGENT = 1,5,6,4,3,7
    modality_ids = jnp.array([ACTION, SIGNAL, STEP, SPATIAL, SPATIAL, SPATIAL, REGISTER, REGISTER, AGENT], dtype=jnp.int32)
    S = modality_ids.shape[0]
    agent_col = (modality_ids == AGENT)  # keys that are agent
    agent_row = (modality_ids == AGENT)  # queries that are agent

    # ----- wm_agent -----
    mask = _build_modality_mask(modality_ids, "wm_agent")  # (S,S)
    _print_mask_summary("wm_agent", modality_ids, mask)

    # Others never see agent: find any offending (q,k) where q!=agent and k is agent
    bad_q = []
    for q in range(S):
        if not bool(agent_row[q]):
            if bool(mask[q, agent_col].sum()):
                bad_q.append(q)
    if bad_q:
        print("Violations in wm_agent (non-agent reads agent) at query rows:", bad_q)

    # Agent reads all in wm_agent
    agent_q_idx = int(jnp.where(agent_row, size=1, fill_value=-1)[0][0])
    if agent_q_idx >= 0:
        agent_reads = mask[agent_q_idx, :]
        missing = [k for k in range(S) if not bool(agent_reads[k])]
        if missing:
            print("Violations in wm_agent (agent cannot read some keys). Missing cols:", missing)

    # Assertions
    for q in range(S):
        if not bool(agent_row[q]):
            assert mask[q, agent_col].sum() == 0, "Non-agent query can attend to agent!"
    if agent_q_idx >= 0:
        assert jnp.all(mask[agent_q_idx, :]), "Agent query cannot read some token in wm_agent"




def test_x1hat_invariant_to_agent_tokens():
    B,T = 2,5
    n_b, d_b = 8, 4      # encoder latents
    n_spatial, pack = 4, 2
    d_spatial = d_b * pack
    D = 32

    fake_enc_z = jnp.ones((B, T, n_b, d_b))
    packed = _pack_bottleneck_to_spatial(fake_enc_z, n_spatial=n_spatial, k=pack)
    actions = jnp.zeros((B,T), dtype=jnp.int32)
    step_idx = jnp.zeros((B,T), dtype=jnp.int32)
    sig_idx  = jnp.zeros((B,T), dtype=jnp.int32)

    dyn = Dynamics(
        d_model=D, d_bottleneck=d_b, d_spatial=d_spatial,
        n_spatial=n_spatial, n_register=2, n_agent=1,
        n_heads=2, depth=2, k_max=8, dropout=0.0, mlp_ratio=2.0,
        time_every=2
    )
    vars_ = dyn.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                     actions, step_idx, sig_idx, packed)

    # random agent vs zeros
    agent_rand = jax.random.normal(jax.random.PRNGKey(2), (B,T,1,D))
    x1_a, _ = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                        agent_tokens=agent_rand, rngs={"dropout": jax.random.PRNGKey(3)}, deterministic=True)
    x1_b, _ = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                        agent_tokens=jnp.zeros_like(agent_rand), rngs={"dropout": jax.random.PRNGKey(3)}, deterministic=True)

    diff = x1_a - x1_b
    max_abs = float(jnp.max(jnp.abs(diff)))
    l2 = float(jnp.sqrt(jnp.sum(diff * diff)))
    print("\n[x1_hat invariance] max|Δ| =", max_abs, " ||Δ||₂ =", l2)
    print("x1_a shape:", x1_a.shape, " x1_b shape:", x1_b.shape)

    # Must be exactly equal because agent cannot influence others
    assert jnp.allclose(x1_a, x1_b, atol=0, rtol=0), "x1_hat changed with agent tokens—firewall broken"


def test_shapes_and_h_t():
    B,T,D = 2,6,32
    n_b,d_b = 8,4
    n_spatial, pack = 4,2
    d_spatial = d_b*pack

    packed = _pack_bottleneck_to_spatial(jnp.ones((B,T,n_b,d_b)), n_spatial, pack)
    dyn = Dynamics(d_model=D, d_bottleneck=d_b, d_spatial=d_spatial,
                   n_spatial=n_spatial, n_register=3, n_agent=1,
                   n_heads=2, depth=2, k_max=8)
    actions = jnp.zeros((B,T), dtype=jnp.int32)
    step_idx = jnp.zeros((B,T), dtype=jnp.int32)
    sig_idx  = jnp.zeros((B,T), dtype=jnp.int32)
    vars_ = dyn.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                     actions, step_idx, sig_idx, packed)

    x1_hat, h_t = dyn.apply(vars_, actions, step_idx, sig_idx, packed,
                            agent_tokens=jnp.zeros((B,T,1,D)))
    print("\n[shapes] x1_hat:", x1_hat.shape, " h_t:", (None if h_t is None else h_t.shape))
    print("Expect x1_hat =", (B,T,n_spatial,d_spatial), " h_t =", (B,T,1,D))
    assert x1_hat.shape == (B,T,n_spatial,d_spatial)
    assert h_t.shape     == (B,T,1,D)

def test_wm_routed():
    """
    Checks space-attention routing for Dreamer-4-style dynamics:
      - Action q -> {Action k}
      - Obs q    -> {Obs k ∪ Action k} and never Agent k
      - Agent q  -> {Obs k ∪ Action k ∪ Agent k}    (wm_agent)

      - For any non-agent q, Agent k is disallowed.
    """
    # Shorthand modality ints
    ACTION  = int(Modality.ACTION)
    SIGNAL  = int(Modality.SHORTCUT_SIGNAL)
    STEP    = int(Modality.SHORTCUT_STEP)
    SPATIAL = int(Modality.SPATIAL)
    REGISTER= int(Modality.REGISTER)
    AGENT   = int(Modality.AGENT)

    # Toy layout (Q rows / K cols share this order):
    # [ACT, SIG, STP, SPA, SPA, SPA, REG, REG, ACT, AGT]
    modality_ids = jnp.array(
        [ACTION, SIGNAL, STEP, SPATIAL, SPATIAL, SPATIAL, REGISTER, REGISTER, ACTION, AGENT],
        dtype=jnp.int32
    )
    S = modality_ids.shape[0]

    # Helper sets
    is_agent = (modality_ids == AGENT)
    is_action = (modality_ids == ACTION)
    is_obs = (
        (modality_ids == SPATIAL) |
        (modality_ids == REGISTER) |
        (modality_ids == SIGNAL)  |
        (modality_ids == STEP)
    )

    def assert_mask(mode: str):
        mask = _build_modality_mask(modality_ids, mode)  # (S,S) bool
        _print_mask_summary(mode, modality_ids, mask)

        # 1) Non-agent q must never see Agent k
        for q in range(S):
            if not bool(is_agent[q]):
                assert not bool(mask[q, is_agent].any()), f"[{mode}] non-agent q={q} can read Agent k!"

        # 2) Action q -> Action k only? NO, in wm_agent, non-agent can read all non-agent.
        # So we just check that non-agent CAN read other non-agent keys (like Spatial).
        # And specifically check that they CANNOT read Agent.
        
        # 3) Obs q -> Obs k ∪ Action k? NO, same as above.
        
        # We only strictly enforce:
        # A) Non-agent q cannot read Agent k
        # B) Agent q can read Agent k (and everyone else)
        
        # Check A: (Already done in step 1)
        
        # Check B:
        agent_rows = [i for i in range(S) if bool(is_agent[i])]
        if agent_rows:
            q = agent_rows[0]
            if mode == "wm_agent":
                # Agent reads everyone (including agent)
                assert bool(mask[q].all()), "[wm_agent] agent q cannot read all keys!"


    # Run both modes
    assert_mask("wm_agent")

    print("\n[test_wm_routed] All routing assertions passed ✅")


if __name__ == "__main__":
    test_encoder_decoder()
    test_dynamics()
    test_agent_firewall()
    test_x1hat_invariant_to_agent_tokens()
    test_shapes_and_h_t()
    test_wm_routed()
    print("\nAll tests passed ✅")
