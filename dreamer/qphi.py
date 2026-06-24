"""Learned perturbation network ``Qphi`` for perturbation matching.

Qphi is a *small* causal network that models the per-frame world-model error
distribution ``p(e | z, t)`` and supplies a learned, content-dependent replacement
for fixed-Gaussian context forcing. It is **not** a second world model and **not** a
diffusion model: it emits an explicit per-frame distribution over the low-dimensional
latent error so we get cheap *exact* likelihood and one-pass sampling inside the
training loop.

Architecture (see ``QphiModelConfig``):

* Backbone: a small causal transformer over the *clean* latent sequence ``z[1:T]``,
  producing per-position conditioning features ``c[t]`` from ``z[<= t]`` (causal mask
  required) plus a noise-level embedding of ``t``.
* Output head: a conditional normalizing flow with a low-rank-plus-diagonal Gaussian
  base ``N(mu, U U^T + diag(s))`` (rank ``r << d_e`` captures the anisotropy). The flow
  is identity-initialised, so the module starts as the conditional anisotropic Gaussian
  and only departs from it (skew / tails) if the data demands.

Conventions
-----------
* ``e`` and ``z`` are in the *unpacked, normalised* latent space the dynamics model
  regresses in, shape ``(B, T, n_latents, d_bottleneck)``; flattened per frame to
  ``d_e = n_latents * d_bottleneck``.
* ``t`` is this repo's *signal* level ``sigma`` (sigma=1 clean, sigma=0 max noise).
* Qphi conditions **only** on clean latents (never on world-model features) and is
  **per-frame** (it does not take other frames' errors as input).
"""
from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

from .configs import QphiModelConfig
from .parallel import MeshRules
from .utils import to_jnp_dtype

LOG_2PI = math.log(2.0 * math.pi)


def _sinusoidal_time_embed(t: jax.Array, dim: int, max_period: float = 10000.0) -> jax.Array:
    """Sinusoidal embedding of a (broadcastable) timestep array.

    Args:
        t: (...) float array of noise levels.
        dim: output embedding dimension.
    Returns:
        (..., dim) embedding.
    """
    t = t.astype(jnp.float32)
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t[..., None] * freqs
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[..., :1])], axis=-1)
    return emb


class _CausalSelfAttention(nnx.Module):
    """Multi-head causal self-attention over the time axis."""

    def __init__(self, dim: int, n_heads: int, *, dtype, param_dtype,
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.dtype = dtype
        self.to_qkv = nnx.Linear(
            dim, 3 * dim, use_bias=False, dtype=dtype, param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('attn')),
            rngs=rngs,
        )
        self.to_out = nnx.Linear(
            dim, dim, use_bias=False, dtype=dtype, param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('attn')),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, T, D)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b t (three n h) -> three b t n h", three=3, n=self.n_heads)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "b t n h -> b t (n h)")
        return self.to_out(out)


class _Block(nnx.Module):
    """Pre-norm transformer block (causal attention + SwiGLU-free MLP)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, *, dtype, param_dtype,
                 mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.norm1 = nnx.RMSNorm(dim, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        self.attn = _CausalSelfAttention(dim, n_heads, dtype=dtype, param_dtype=param_dtype,
                                         mesh_rules=mesh_rules, rngs=rngs)
        self.norm2 = nnx.RMSNorm(dim, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, use_bias=False, dtype=dtype, param_dtype=param_dtype,
                              kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')),
                              rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, use_bias=False, dtype=dtype, param_dtype=param_dtype,
                              kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')),
                              rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(jax.nn.silu(self.fc1(self.norm2(x))))
        return x


class _Backbone(nnx.Module):
    """Small causal transformer producing per-position conditioning features c[t]."""

    def __init__(self, cfg: QphiModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)
        self.d_model = cfg.d_model
        self.in_proj = nnx.Linear(
            cfg.d_e, cfg.d_model, use_bias=False, dtype=dtype, param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')),
            rngs=rngs,
        )
        self.time_proj = nnx.Linear(
            cfg.d_model, cfg.d_model, use_bias=True, dtype=dtype, param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.normal(0.02), mesh_rules('mlp')),
            rngs=rngs,
        )
        self.blocks = nnx.List([
            _Block(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, dtype=dtype, param_dtype=param_dtype,
                   mesh_rules=mesh_rules, rngs=rngs)
            for _ in range(cfg.depth)
        ])
        self.out_norm = nnx.RMSNorm(cfg.d_model, dtype=jnp.float32, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, z_flat: jax.Array, t: jax.Array) -> jax.Array:
        # z_flat: (B, T, d_e); t: (B,) or (B, T)
        B, T, _ = z_flat.shape
        if t.ndim == 1:
            t = jnp.broadcast_to(t[:, None], (B, T))
        h = self.in_proj(z_flat)
        temb = _sinusoidal_time_embed(t, self.d_model).astype(h.dtype)
        h = h + self.time_proj(temb)
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)


class _CouplingLayer(nnx.Module):
    """RealNVP-style affine coupling layer, conditioned on c, identity-initialised."""

    def __init__(self, d_e: int, hidden: int, cond_dim: int, mask: jax.Array, clamp: float,
                 *, dtype, param_dtype, mesh_rules: MeshRules, rngs: nnx.Rngs):
        # mask is a constant (non-trained) buffer: 1 = "fixed" dims that condition the
        # transform, 0 = "transformed" dims. Stored as a non-Param Variable.
        self.mask = nnx.Variable(mask.astype(jnp.float32))
        self.clamp = clamp
        self.net1 = nnx.Linear(d_e + cond_dim, hidden, use_bias=True, dtype=dtype, param_dtype=param_dtype,
                               kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')),
                               rngs=rngs)
        self.net2 = nnx.Linear(hidden, hidden, use_bias=True, dtype=dtype, param_dtype=param_dtype,
                               kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), mesh_rules('mlp')),
                               rngs=rngs)
        # Zero-init final layer => log_scale=0, shift=0 => identity transform at init.
        self.out = nnx.Linear(hidden, 2 * d_e, use_bias=True, dtype=dtype, param_dtype=param_dtype,
                              kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
                              bias_init=nnx.initializers.zeros, rngs=rngs)

    def _params(self, x: jax.Array, c: jax.Array) -> Tuple[jax.Array, jax.Array]:
        m = self.mask.value
        x_masked = x * m
        h = jnp.concatenate([x_masked, c], axis=-1)
        h = jax.nn.silu(self.net1(h))
        h = jax.nn.silu(self.net2(h))
        log_s, shift = jnp.split(self.out(h), 2, axis=-1)
        # Bounded log-scale (tanh) so the jacobian stays well-conditioned; 0 at init.
        log_s = self.clamp * jnp.tanh(log_s.astype(jnp.float32))
        shift = shift.astype(jnp.float32)
        # Only the un-fixed (mask==0) dims are transformed.
        log_s = log_s * (1.0 - m)
        shift = shift * (1.0 - m)
        return log_s, shift

    def forward(self, x: jax.Array, c: jax.Array) -> Tuple[jax.Array, jax.Array]:
        log_s, shift = self._params(x, c)
        y = x * jnp.exp(log_s) + shift
        return y, jnp.sum(log_s, axis=-1)

    def inverse(self, y: jax.Array, c: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # Conditioner reads only fixed dims (unchanged by the transform), so it is exact.
        log_s, shift = self._params(y, c)
        x = (y - shift) * jnp.exp(-log_s)
        return x, -jnp.sum(log_s, axis=-1)


class _CouplingFlow(nnx.Module):
    """Stack of alternating-mask affine coupling layers."""

    def __init__(self, cfg: QphiModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)
        d_e = cfg.d_e
        base = (jnp.arange(d_e) % 2).astype(jnp.float32)  # alternating parity mask
        self.layers = nnx.List([
            _CouplingLayer(
                d_e, cfg.flow_hidden, cfg.d_model,
                mask=base if (i % 2 == 0) else (1.0 - base),
                clamp=cfg.flow_logscale_clamp,
                dtype=dtype, param_dtype=param_dtype, mesh_rules=mesh_rules, rngs=rngs,
            )
            for i in range(cfg.n_flow_layers)
        ])

    def forward(self, x: jax.Array, c: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """base -> data; returns (y, sum log|det dy/dx|)."""
        logdet = jnp.zeros(x.shape[:-1], dtype=jnp.float32)
        for layer in self.layers:
            x, ld = layer.forward(x, c)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, y: jax.Array, c: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """data -> base; returns (x, sum log|det dx/dy|)."""
        logdet = jnp.zeros(y.shape[:-1], dtype=jnp.float32)
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y, c)
            logdet = logdet + ld
        return y, logdet


class Qphi(nnx.Module):
    """Learned per-frame perturbation distribution ``p(e | z, t)``.

    Interface (all in unpacked, normalised latent space; e/z: (B, T, n_latents, d_b)):
        ``forward_cond(z, t) -> c``          conditioning features (B, T, d_model)
        ``log_prob(e, z, t) -> logp``        exact per-frame log-density (B, T)
        ``sample(z, t, rng) -> pert``        one-pass sample (B, T, n_latents, d_b)
    """

    def __init__(self, cfg: QphiModelConfig, *, mesh_rules: MeshRules, rngs: nnx.Rngs):
        self.cfg = cfg
        self.dtype = to_jnp_dtype(cfg.dtype)
        param_dtype = to_jnp_dtype(cfg.param_dtype)
        self.d_e = cfg.d_e
        self.type = cfg.type
        self.s_floor = cfg.s_floor
        self.n_latents = cfg.n_latents
        self.is_iso = (cfg.type == "gaussian_iso")
        self.has_flow = (cfg.type == "flow")
        self.rank = cfg.rank if cfg.type in ("gaussian_lowrank", "flow") else 0

        self.backbone = _Backbone(cfg, mesh_rules=mesh_rules, rngs=rngs)

        # Mean head: zero-init so the base starts mean-zero (matches the N(0, I) warmup
        # start); it learns the error bias from data.
        self.mu_head = nnx.Linear(
            cfg.d_model, cfg.d_e, use_bias=True, dtype=self.dtype, param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
            bias_init=nnx.initializers.zeros, rngs=rngs,
        )

        # Diagonal variance head. Bias init ~ softplus^{-1}(1) so s ~= 1 at init
        # (unit variance in normalised latent space == the fixed-Gaussian start).
        s_init = float(math.log(math.expm1(1.0)))  # softplus(s_init) == 1
        diag_out = 1 if self.is_iso else cfg.d_e
        self.logdiag_head = nnx.Linear(
            cfg.d_model, diag_out, use_bias=True, dtype=self.dtype, param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.zeros, mesh_rules('mlp')),
            bias_init=nnx.initializers.constant(s_init), rngs=rngs,
        )

        # Low-rank factor head. We CANNOT zero-init U: U=0 is an exact saddle of the
        # Gaussian log-likelihood (d logp / dU = 0 there), so gradient descent would never
        # develop any anisotropy. Instead use a *tiny* fan-in-scaled init: the initial
        # U U^T contribution is ~ rank * 1e-3 (a few % of the unit diagonal, so the base
        # still starts as ~N(0, I)) while the gradient is non-zero so anisotropy can emerge.
        if self.rank > 0:
            self.U_head = nnx.Linear(
                cfg.d_model, cfg.d_e * self.rank, use_bias=True, dtype=self.dtype, param_dtype=param_dtype,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.variance_scaling(1e-3, "fan_in", "normal"), mesh_rules('mlp')),
                bias_init=nnx.initializers.zeros, rngs=rngs,
            )

        if self.has_flow:
            self.flow = _CouplingFlow(cfg, mesh_rules=mesh_rules, rngs=rngs)

    # ---- helpers ----

    @staticmethod
    def _flatten(x: jax.Array) -> jax.Array:
        return rearrange(x, "b t n d -> b t (n d)")

    def _unflatten(self, x: jax.Array) -> jax.Array:
        return rearrange(x, "b t (n d) -> b t n d", n=self.n_latents)

    def forward_cond(self, z: jax.Array, t: jax.Array) -> jax.Array:
        """Per-position conditioning features c[t] from z[<= t] and a t-embedding."""
        return self.backbone(self._flatten(z), t)

    def _base_params(self, c: jax.Array):
        """Returns (mu, s, U) in float32. s is per-dim variance; U may be None."""
        mu = self.mu_head(c).astype(jnp.float32)
        s = jax.nn.softplus(self.logdiag_head(c).astype(jnp.float32)) + self.s_floor
        if self.is_iso:
            s = jnp.broadcast_to(s, mu.shape)  # (B, T, d_e) isotropic
            U = None
        elif self.rank > 0:
            U = self.U_head(c).astype(jnp.float32)
            U = rearrange(U, "b t (d r) -> b t d r", r=self.rank)
        else:
            U = None
        return mu, s, U

    def _base_log_prob(self, x: jax.Array, mu: jax.Array, s: jax.Array, U) -> jax.Array:
        """Exact log N(x; mu, diag(s) + U U^T). Shapes: x,mu,s (B,T,d_e); U (B,T,d_e,r)."""
        xc = (x - mu).astype(jnp.float32)
        D = self.d_e
        if U is None:
            # Diagonal Gaussian.
            logdet = jnp.sum(jnp.log(s), axis=-1)
            quad = jnp.sum(xc * xc / s, axis=-1)
            return -0.5 * (D * LOG_2PI + logdet + quad)
        # Low-rank-plus-diagonal via matrix-determinant lemma + Woodbury identity.
        r = U.shape[-1]
        Dinv = 1.0 / s
        # M = I_r + U^T diag(Dinv) U   (B, T, r, r)
        M = jnp.eye(r) + jnp.einsum("...di,...d,...dj->...ij", U, Dinv, U)
        _, logdetM = jnp.linalg.slogdet(M)
        logdet = jnp.sum(jnp.log(s), axis=-1) + logdetM
        a = Dinv * xc                                  # (B, T, d_e)
        w = jnp.einsum("...d,...dr->...r", a, U)       # U^T Dinv xc
        y = jnp.linalg.solve(M, w[..., None])[..., 0]  # M^{-1} w
        quad = jnp.sum(a * xc, axis=-1) - jnp.sum(w * y, axis=-1)
        return -0.5 * (D * LOG_2PI + logdet + quad)

    def log_prob(self, e: jax.Array, z: jax.Array, t: jax.Array) -> jax.Array:
        """Exact per-frame log-density of error ``e`` under ``p(. | z, t)``.

        Args:
            e: (B, T, n_latents, d_bottleneck) residual target (stop-gradient upstream).
            z: (B, T, n_latents, d_bottleneck) clean latents (conditioner).
            t: (B,) or (B, T) signal level sigma.
        Returns:
            (B, T) log-density.
        """
        c = self.forward_cond(z, t)
        mu, s, U = self._base_params(c)
        e_flat = self._flatten(e).astype(jnp.float32)
        if self.has_flow:
            e_base, logdet_inv = self.flow.inverse(e_flat, c)
            return self._base_log_prob(e_base, mu, s, U) + logdet_inv
        return self._base_log_prob(e_flat, mu, s, U)

    def sample(self, z: jax.Array, t: jax.Array, rng: jax.Array) -> jax.Array:
        """One-pass sample of the perturbation conditioned on clean latents ``z``.

        Returns ``pert`` shaped like ``z`` (B, T, n_latents, d_bottleneck).
        """
        c = self.forward_cond(z, t)
        mu, s, U = self._base_params(c)
        B, T, _ = mu.shape
        k_d, k_r = jax.random.split(rng)
        eps_d = jax.random.normal(k_d, (B, T, self.d_e), dtype=jnp.float32)
        base = mu + jnp.sqrt(s) * eps_d
        if U is not None:
            eps_r = jax.random.normal(k_r, (B, T, self.rank), dtype=jnp.float32)
            base = base + jnp.einsum("...dr,...r->...d", U, eps_r)
        out = self.flow.forward(base, c)[0] if self.has_flow else base
        return self._unflatten(out.astype(self.dtype))

    # ---- diagnostics ----

    def expected_trace(self, z: jax.Array, t: jax.Array) -> jax.Array:
        """Mean trace(Sigma) of the base covariance over (B, T) (anti-collapse monitor)."""
        c = self.forward_cond(z, t)
        _, s, U = self._base_params(c)
        tr = jnp.sum(s, axis=-1)
        if U is not None:
            tr = tr + jnp.sum(U * U, axis=(-1, -2))
        return jnp.mean(tr)

    def rank_energy(self, z: jax.Array, t: jax.Array) -> jax.Array:
        """Anisotropy spectrum: mean eigenvalues of ``U U^T`` over (B, T), descending.

        These are the squared singular values of ``U`` (the columns of ``U`` themselves are
        not identifiable up to rotation, so their raw norms are not meaningful). A few
        dominant entries == anisotropy emerged in a few content-dependent directions; a
        flat spectrum == it did not. Returns ``(r,)`` (or a scalar 0 when there is no
        low-rank factor).
        """
        c = self.forward_cond(z, t)
        _, _, U = self._base_params(c)
        if U is None:
            return jnp.zeros((max(self.rank, 1),))
        sv = jnp.linalg.svd(U, compute_uv=False)  # (B, T, r) singular values, descending
        return jnp.mean(sv ** 2, axis=(0, 1))      # (r,) eigenvalues of U U^T
