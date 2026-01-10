"""
Quick test script for mean flow forcing implementation.

This script verifies that:
1. sample_r_t_for_meanflow() works correctly
2. SinusoidalEmbedding produces valid outputs
3. Dynamics model accepts continuous conditioning
4. meanflow_forcing_step() runs without errors

Run this before full training to catch any basic issues.
"""
import jax
import jax.numpy as jnp
from flax import nnx

from dreamer.models import Dynamics, SinusoidalEmbedding
from dreamer.configs import DynamicsModelConfig
from dreamer.training import sample_r_t_for_meanflow, meanflow_forcing_step
from dreamer.parallel import build_parallel


def test_sampling():
    """Test that (r, t) sampling works correctly."""
    print("\n=== Testing sample_r_t_for_meanflow() ===")
    rng = jax.random.PRNGKey(0)
    k_max = 8
    r, t, delta = sample_r_t_for_meanflow(rng, (4, 16), k_max)

    # Verify constraints
    assert jnp.all(r < t), "r must be less than t"
    assert jnp.allclose(delta, t - r), "delta must equal t - r"
    assert jnp.all((r >= 0) & (r <= 1)), "r must be in [0, 1]"
    assert jnp.all((t >= 0) & (t <= 1)), "t must be in [0, 1]"

    print(f"✓ Sampling test passed")
    print(f"  Sample r values: {r[0, :5]}")
    print(f"  Sample t values: {t[0, :5]}")
    print(f"  Sample delta values: {delta[0, :5]}")


def test_sinusoidal_embedding():
    """Test that sinusoidal embeddings work correctly."""
    print("\n=== Testing SinusoidalEmbedding ===")
    d_model = 128
    emb = SinusoidalEmbedding(d_model)

    x = jnp.array([[0.0, 0.25, 0.5, 0.75, 1.0]])  # (1, 5)
    out = emb(x)

    assert out.shape == (1, 5, d_model), f"Expected shape (1, 5, {d_model}), got {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "Output should be finite"

    print(f"✓ Sinusoidal embedding test passed")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{jnp.min(out):.3f}, {jnp.max(out):.3f}]")


def test_dynamics_continuous_mode():
    """Test that Dynamics model accepts continuous conditioning."""
    print("\n=== Testing Dynamics with continuous conditioning ===")

    # Setup
    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        # Create tiny model for testing
        cfg = DynamicsModelConfig(
            d_model=64,
            d_bottleneck=8,
            action_dim=4,
            depth=2,
            n_heads=2,
            n_kv_heads=1,
            packing_factor=2,
            n_register=2,
            k_max=8,
        )

        # Initialize model
        rng = jax.random.PRNGKey(42)
        dynamics = Dynamics(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(rng))

        # Create dummy inputs
        B, T, S, D = 2, 4, 8, cfg.d_bottleneck * cfg.packing_factor
        packed_enc_tokens = jnp.ones((B, T, S // cfg.packing_factor, D))
        actions = jnp.zeros((B, T), dtype=jnp.int32)
        r = jnp.ones((B, T)) * 0.3
        t = jnp.ones((B, T)) * 0.7

        # Test forward pass with continuous conditioning
        output, _ = dynamics(
            actions,
            jnp.zeros((B, T), dtype=jnp.int32),  # dummy step_indices
            jnp.zeros((B, T), dtype=jnp.int32),  # dummy tau_indices
            packed_enc_tokens,
            r_continuous=r,
            t_continuous=t,
            deterministic=True,
        )

        assert output.shape == packed_enc_tokens.shape, \
            f"Expected output shape {packed_enc_tokens.shape}, got {output.shape}"
        assert jnp.all(jnp.isfinite(output)), "Output should be finite"

        print(f"✓ Dynamics continuous mode test passed")
        print(f"  Input shape: {packed_enc_tokens.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{jnp.min(output):.3f}, {jnp.max(output):.3f}]")


def test_meanflow_forcing_step():
    """Test that meanflow_forcing_step() runs without errors."""
    print("\n=== Testing meanflow_forcing_step() ===")

    # Setup
    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        # Create tiny model for testing
        cfg = DynamicsModelConfig(
            d_model=64,
            d_bottleneck=8,
            action_dim=4,
            depth=2,
            n_heads=2,
            n_kv_heads=1,
            packing_factor=2,
            n_register=2,
            k_max=8,
        )

        # Initialize model
        rng = jax.random.PRNGKey(123)
        dynamics = Dynamics(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(rng))

        # Create dummy inputs
        B, T, S, D = 2, 4, 8, cfg.d_bottleneck * cfg.packing_factor
        latents = jax.random.normal(rng, (B, T, S // cfg.packing_factor, D))
        actions = jnp.zeros((B, T), dtype=jnp.int32)

        # Run training step
        rng_step = jax.random.PRNGKey(456)
        losses, aux = meanflow_forcing_step(
            dynamics_model=dynamics,
            actions=actions,
            latents=latents,
            rng=rng_step,
            k_max=cfg.k_max,
            agent_tokens=None,
        )

        # Verify outputs
        assert 'total' in losses, "losses should contain 'total'"
        assert 'meanflow' in losses, "losses should contain 'meanflow'"
        assert 'meanflow_mse' in aux, "aux should contain 'meanflow_mse'"
        assert jnp.isfinite(losses['total']), "Loss should be finite"

        print(f"✓ meanflow_forcing_step() test passed")
        print(f"  Total loss: {losses['total']:.6f}")
        print(f"  Meanflow MSE: {aux['meanflow_mse']:.6f}")


def test_meanflow_sampler():
    """Test that meanflow sampler generates valid videos."""
    print("\n=== Testing meanflow sampler ===")

    from dreamer.generation import next_latent_meanflow

    # Setup
    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        # Create tiny model for testing
        cfg = DynamicsModelConfig(
            d_model=64,
            d_bottleneck=8,
            action_dim=4,
            depth=2,
            n_heads=2,
            n_kv_heads=1,
            packing_factor=2,
            n_register=2,
            k_max=8,
            forcing_type="meanflow",
        )

        # Initialize model
        rng = jax.random.PRNGKey(789)
        dynamics = Dynamics(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(rng))

        # Test 1-step generation (direct)
        B, S, D = 2, 8, cfg.d_bottleneck * cfg.packing_factor
        latent_shape = (B, 1, S // cfg.packing_factor, D)
        action = jnp.zeros((B, 1), dtype=jnp.int32)

        rng_sample = jax.random.PRNGKey(999)
        latent, h_last, caches, rng_out = next_latent_meanflow(
            dynamics=dynamics,
            num_steps=1,  # 1-step direct generation
            action=action,
            latent_shape=latent_shape,
            rng=rng_sample,
            caches=None,
        )

        # Verify outputs
        assert latent.shape == latent_shape, f"Expected shape {latent_shape}, got {latent.shape}"
        assert jnp.all(jnp.isfinite(latent)), "Latent should be finite"

        print(f"✓ Meanflow sampler test passed (1-step generation)")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Latent range: [{jnp.min(latent):.3f}, {jnp.max(latent):.3f}]")

        # Test 4-step generation (multi-step)
        latent_4step, h_last_4, caches_4, rng_out_4 = next_latent_meanflow(
            dynamics=dynamics,
            num_steps=4,  # 4-step refinement
            action=action,
            latent_shape=latent_shape,
            rng=rng_sample,
            caches=None,
        )

        assert latent_4step.shape == latent_shape
        assert jnp.all(jnp.isfinite(latent_4step))

        print(f"✓ Meanflow sampler test passed (4-step generation)")
        print(f"  Latent shape: {latent_4step.shape}")
        print(f"  Latent range: [{jnp.min(latent_4step):.3f}, {jnp.max(latent_4step):.3f}]")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Mean Flow Forcing - Basic Functionality Tests")
    print("=" * 60)

    try:
        test_sampling()
        test_sinusoidal_embedding()
        test_dynamics_continuous_mode()
        test_meanflow_forcing_step()
        test_meanflow_sampler()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nYou can now train with mean flow forcing by setting:")
        print("  dynamics.forcing_type=meanflow")
        print("\nExample:")
        print("  python scripts/train_dynamics.py dynamics.forcing_type=meanflow")
        print("\nEvaluation will automatically use the meanflow sampler for meanflow-trained models.")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Test failed!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
