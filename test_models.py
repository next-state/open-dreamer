
import jax
import jax.numpy as jnp
from dreamer.models import Encoder, Decoder, Dynamics, TaskEmbedder, PolicyHeadMTP, RewardHeadMTP

def test_models():
    # Encoder
    enc = Encoder(
        d_model=64, n_latents=4, n_patches=16, n_heads=2, n_kv_heads=1,
        depth=2, d_bottleneck=32, dropout_rate=0.0
    )
    patches = jnp.zeros((1, 8, 16, 64)) # B, T, Np, D
    rng = jax.random.PRNGKey(0)
    vars_enc = enc.init(rng, patches, deterministic=True)
    out, _ = enc.apply(vars_enc, patches, rngs={'mae': rng}, deterministic=True)
    print("Encoder output shape:", out.shape)

    # Decoder
    dec = Decoder(
        d_model=64, n_heads=2, n_kv_heads=1, depth=2, n_latents=4,
        n_patches=16, d_patch=48, dropout_rate=0.0
    )
    z = jnp.zeros((1, 8, 4, 32)) # B, T, Nl, Db
    vars_dec = dec.init(rng, z, deterministic=True)
    out = dec.apply(vars_dec, z, deterministic=True)
    print("Decoder output shape:", out.shape)

    # Dynamics
    dyn = Dynamics(
        d_model=128, d_bottleneck=32, d_spatial=64, n_spatial=2,
        n_register=2, n_agent=1, n_heads=2, n_kv_heads=1, depth=2,
        k_max=4, dropout_rate=0.0
    )
    actions = jnp.zeros((1, 8), dtype=jnp.int32)
    step_idxs = jnp.zeros((1, 8), dtype=jnp.int32)
    sig_idxs = jnp.zeros((1, 8), dtype=jnp.int32)
    packed = jnp.zeros((1, 8, 2, 64))
    vars_dyn = dyn.init(rng, actions, step_idxs, sig_idxs, packed, deterministic=True)
    x1_hat, h_t = dyn.apply(vars_dyn, actions, step_idxs, sig_idxs, packed, deterministic=True)
    print("Dynamics output shape:", x1_hat.shape)

if __name__ == "__main__":
    test_models()
