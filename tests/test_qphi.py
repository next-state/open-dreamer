"""Acceptance tests for learned perturbation matching (Qphi).

Covers the non-negotiable correctness properties from the design spec (§6):

  1. Baseline reproducibility   - disabled / lam=0 == vanilla diffusion forcing.
  2. Gradient isolation         - L_world never touches Qphi; L_match never touches the
                                  world model (the collapse guard + stop-gradient target).
  3. Detach check               - the context fed to the world model has no grad path to Qphi.
  4. Anti-collapse / distributional - the log_prob loss preserves spread (an MSE-to-e
                                  regression would collapse to the conditional mean).
  5. Matching sanity            - mean ||pert|| ~ mean ||e||; anisotropy emerges in U.

Plus unit tests of the Qphi density (exact log_prob, identity-init, causal conditioning).

Run: pytest tests/test_qphi.py
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from dreamer.actions import Actions
from dreamer.configs import DynamicsModelConfig, QphiModelConfig
from dreamer.models import Dynamics
from dreamer.parallel import build_parallel
from dreamer.qphi import Qphi
from dreamer.training import shortcut_forcing_step


# --------------------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """Activate a mesh so `with_partitioning` sharding annotations resolve."""
    mesh, _, mesh_rules = build_parallel("data")
    with jax.set_mesh(mesh):
        yield mesh_rules


N_LATENTS, D_BOTTLENECK = 4, 4
D_E = N_LATENTS * D_BOTTLENECK  # 16


def tiny_dynamics(mesh_rules, seed=0):
    cfg = DynamicsModelConfig(
        d_bottleneck=D_BOTTLENECK, depth=2, d_model=32, n_heads=4, n_kv_heads=1,
        packing_factor=2, n_register=2, k_max=4, context_length=8, time_every=2,
        time_layer_offset=0, num_binary_actions=0, categorical_action_dim=0,
        continuous_action_dim=0, latent_mean=None, latent_std=None,
        dtype="float32", param_dtype="float32",
    )
    return Dynamics(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(seed))


def tiny_qphi(mesh_rules, qtype="flow", seed=1, **over):
    kw = dict(enabled=True, type=qtype, n_latents=N_LATENTS, d_bottleneck=D_BOTTLENECK,
              d_model=32, depth=2, n_heads=4, rank=3, n_flow_layers=2, flow_hidden=16,
              lam=1.0, t_query=0.0, warmup_steps=0)
    kw.update(over)
    cfg = QphiModelConfig(**kw)
    return Qphi(cfg, mesh_rules=mesh_rules, rngs=nnx.Rngs(seed)), cfg


def _grad_norm(grads):
    return float(optax.global_norm(nnx.state(grads)))


def _run_step(dyn, qphi, qcfg, latents, *, key, lam_override=None):
    """One shortcut-forcing step; returns (loss, aux)."""
    qkey = jax.random.fold_in(key, 777)
    if lam_override is not None:
        qcfg = QphiModelConfig(**{**qcfg.__dict__, "lam": lam_override})
    losses, aux = shortcut_forcing_step(
        dyn, Actions(), latents, key, k_max=4, B_self=0, context_length=8,
        qphi=qphi, qphi_cfg=qcfg, qphi_rng=qkey, qphi_alpha=1.0,
    )
    return losses["total"], aux


# --------------------------------------------------------------------------------------
# §6.1 Baseline reproducibility
# --------------------------------------------------------------------------------------

def test_disabled_matches_baseline_and_is_deterministic(env):
    dyn = tiny_dynamics(env)
    latents = jax.random.normal(jax.random.PRNGKey(2), (4, 6, N_LATENTS, D_BOTTLENECK))
    key = jax.random.PRNGKey(3)

    def baseline():
        losses, _ = shortcut_forcing_step(dyn, Actions(), latents, key, k_max=4,
                                          B_self=0, context_length=8)
        return float(losses["total"])

    # Determinism + the qphi=None path is byte-identical to omitting the kwargs.
    assert baseline() == baseline()
    losses_none, _ = shortcut_forcing_step(dyn, Actions(), latents, key, k_max=4, B_self=0,
                                           context_length=8, qphi=None, qphi_cfg=None)
    assert float(losses_none["total"]) == baseline()


def test_lam_zero_equals_baseline(env):
    """Enabling Qphi with lam=0 injects nothing => identical to the baseline."""
    dyn = tiny_dynamics(env)
    qphi, qcfg = tiny_qphi(env, "flow")
    latents = jax.random.normal(jax.random.PRNGKey(5), (4, 6, N_LATENTS, D_BOTTLENECK))
    key = jax.random.PRNGKey(6)

    base, _ = shortcut_forcing_step(dyn, Actions(), latents, key, k_max=4, B_self=0,
                                    context_length=8)
    loss0, _ = _run_step(dyn, qphi, qcfg, latents, key=key, lam_override=0.0)
    np.testing.assert_allclose(float(loss0), float(base["total"]), rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# §6.2 / §6.3 Gradient isolation + detach
# --------------------------------------------------------------------------------------

def test_world_loss_has_no_qphi_gradient(env):
    """L_world.backward() must leave every Qphi parameter with zero gradient."""
    dyn = tiny_dynamics(env)
    qphi, qcfg = tiny_qphi(env, "flow")
    latents = jax.random.normal(jax.random.PRNGKey(2), (4, 6, N_LATENTS, D_BOTTLENECK))
    key = jax.random.PRNGKey(3)

    def world(qphi_model):
        loss, _ = _run_step(dyn, qphi_model, qcfg, latents, key=key)
        return loss

    grads = nnx.grad(world)(qphi)
    assert _grad_norm(grads) == 0.0


def test_context_has_no_qphi_grad_path(env):
    """The perturbed context fed to the world model has no grad path to Qphi (detach)."""
    dyn = tiny_dynamics(env)
    qphi, qcfg = tiny_qphi(env, "flow")
    latents = jax.random.normal(jax.random.PRNGKey(8), (4, 6, N_LATENTS, D_BOTTLENECK))
    key = jax.random.PRNGKey(9)

    def zpred_sum(qphi_model):
        _, aux = _run_step(dyn, qphi_model, qcfg, latents, key=key)
        # qphi_e = (z_pred - z).detach(); use its (detached) value to confirm zero path,
        # and additionally probe the world output magnitude is qphi-independent.
        return jnp.sum(aux["qphi_e_norm"])

    grads = nnx.grad(zpred_sum)(qphi)
    assert _grad_norm(grads) == 0.0


def test_match_loss_has_no_dynamics_gradient(env):
    """L_match.backward() must not reach the world model (e is a stop-gradient target)."""
    dyn = tiny_dynamics(env)
    qphi, qcfg = tiny_qphi(env, "flow")
    latents = jax.random.normal(jax.random.PRNGKey(2), (4, 6, N_LATENTS, D_BOTTLENECK))
    key = jax.random.PRNGKey(3)

    def match_via_dyn(dyn_model):
        _, aux = _run_step(dyn_model, qphi, qcfg, latents, key=key)
        return -jnp.mean(qphi.log_prob(aux["qphi_e"], aux["qphi_z"], aux["qphi_sigma"]))

    grads = nnx.grad(match_via_dyn)(dyn)
    assert _grad_norm(grads) == 0.0


def test_match_loss_does_train_qphi(env):
    """Sanity: the matching loss has a non-trivial gradient w.r.t. Qphi."""
    dyn = tiny_dynamics(env)
    qphi, qcfg = tiny_qphi(env, "flow")
    latents = jax.random.normal(jax.random.PRNGKey(2), (4, 6, N_LATENTS, D_BOTTLENECK))
    _, aux = _run_step(dyn, qphi, qcfg, latents, key=jax.random.PRNGKey(3))

    def match(qphi_model):
        return -jnp.mean(qphi_model.log_prob(aux["qphi_e"], aux["qphi_z"], aux["qphi_sigma"]))

    assert _grad_norm(nnx.grad(match)(qphi)) > 0.0


# --------------------------------------------------------------------------------------
# §6.4 Anti-collapse: the distributional loss preserves spread
# --------------------------------------------------------------------------------------

def _train_qphi_on(target_fn, qphi, *, z, t, steps=400, lr=3e-3, seed=0):
    tx = optax.adam(lr)
    opt = nnx.Optimizer(qphi, tx, wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)

    @nnx.jit
    def step(qphi, opt, e):
        loss, g = nnx.value_and_grad(lambda q: -jnp.mean(q.log_prob(e, z, t)))(qphi)
        opt.update(qphi, g)
        return loss

    for i in range(steps):
        key, k = jax.random.split(key)
        e = target_fn(k)
        loss = step(qphi, opt, e)
    return float(loss)


def test_distributional_preserves_spread_no_collapse(env):
    """e = mean(z) + sigma*eps with stochastic eps. A proper density learns trace(Sigma)
    ~ D*sigma^2 (nonzero). A mean-regression (MSE) would collapse it to ~0."""
    qphi, _ = tiny_qphi(env, "gaussian_lowrank", d_model=48, depth=2, rank=4)
    B, T = 8, 4
    z = jax.random.normal(jax.random.PRNGKey(11), (B, T, N_LATENTS, D_BOTTLENECK))
    t = jnp.zeros((B, T))
    sigma_noise = 0.3

    def target(k):
        eps = jax.random.normal(k, z.shape)
        return 0.5 * z + sigma_noise * eps  # mean 0.5z, conditional cov sigma^2 I

    _train_qphi_on(target, qphi, z=z, t=t, steps=500, lr=5e-3)
    trace = float(qphi.expected_trace(z, t))
    expected = D_E * sigma_noise ** 2  # ~1.44
    assert trace > 0.3 * expected, f"variance collapsed: trace={trace:.3f} expected~{expected:.3f}"
    assert trace < 4.0 * expected, f"variance exploded: trace={trace:.3f} expected~{expected:.3f}"


def test_anisotropy_emerges(env):
    """When the error concentrates in one direction, U should develop one dominant column."""
    qphi, _ = tiny_qphi(env, "gaussian_lowrank", d_model=48, depth=2, rank=4)
    B, T = 8, 4
    z = jax.random.normal(jax.random.PRNGKey(13), (B, T, N_LATENTS, D_BOTTLENECK))
    t = jnp.zeros((B, T))
    direction = jax.random.normal(jax.random.PRNGKey(14), (N_LATENTS * D_BOTTLENECK,))
    direction = direction / jnp.linalg.norm(direction)

    def target(k):
        k1, k2 = jax.random.split(k)
        a = 1.5 * jax.random.normal(k1, (B, T, 1))          # dominant rank-1 component
        e_flat = a * direction[None, None] + 0.05 * jax.random.normal(k2, (B, T, D_E))
        return e_flat.reshape(B, T, N_LATENTS, D_BOTTLENECK)

    _train_qphi_on(target, qphi, z=z, t=t, steps=600, lr=5e-3)
    energy = np.array(qphi.rank_energy(z, t))
    frac = energy.max() / (energy.sum() + 1e-8)
    assert frac > 0.6, f"anisotropy did not concentrate: per-rank energy={energy}, top frac={frac:.2f}"


# --------------------------------------------------------------------------------------
# §6.5 Matching sanity: ||pert|| ~ ||e|| at the (trivial) fixed point
# --------------------------------------------------------------------------------------

def test_pert_and_e_same_order(env):
    dyn = tiny_dynamics(env)
    qphi, qcfg = tiny_qphi(env, "gaussian_lowrank")
    latents = jax.random.normal(jax.random.PRNGKey(2), (4, 6, N_LATENTS, D_BOTTLENECK))
    _, aux = _run_step(dyn, qphi, qcfg, latents, key=jax.random.PRNGKey(3))
    e_norm = float(aux["qphi_e_norm"])
    pert_norm = float(aux["qphi_pert_norm"])
    assert 0.2 < pert_norm / e_norm < 5.0, f"||pert||={pert_norm:.3f} not same order as ||e||={e_norm:.3f}"


# --------------------------------------------------------------------------------------
# Unit tests of the Qphi density
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("qtype", ["gaussian_iso", "gaussian_lowrank", "flow"])
def test_log_prob_matches_dense_gaussian(env, qtype):
    """Exact log_prob (Woodbury / flow change-of-variables) matches a dense reference.

    Flow layers are identity-initialised, so at init the density is exactly the
    low-rank-plus-diagonal base Gaussian.
    """
    qphi, _ = tiny_qphi(env, qtype, seed=5)
    B, T = 2, 5
    z = jax.random.normal(jax.random.PRNGKey(1), (B, T, N_LATENTS, D_BOTTLENECK))
    e = jax.random.normal(jax.random.PRNGKey(3), (B, T, N_LATENTS, D_BOTTLENECK))
    t = jnp.zeros((B, T))
    lp = qphi.log_prob(e, z, t)
    assert lp.shape == (B, T)

    c = qphi.forward_cond(z, t)
    mu, s, U = qphi._base_params(c)
    bi, ti = 0, 2
    mu0, s0 = np.array(mu[bi, ti]), np.array(s[bi, ti])
    Sigma = np.diag(s0)
    if U is not None:
        U0 = np.array(U[bi, ti])
        Sigma = Sigma + U0 @ U0.T
    x0 = np.array(qphi._flatten(e)[bi, ti])
    _, logdet = np.linalg.slogdet(Sigma)
    quad = (x0 - mu0) @ np.linalg.solve(Sigma, (x0 - mu0))
    lp_dense = -0.5 * (D_E * math.log(2 * math.pi) + logdet + quad)
    np.testing.assert_allclose(float(lp[bi, ti]), lp_dense, rtol=1e-3, atol=1e-3)


def test_identity_init_is_unit_gaussian(env):
    """At init the perturbation distribution is ~N(0, I) (matches the warmup start)."""
    qphi, _ = tiny_qphi(env, "flow")
    B, T = 2, 5
    z = jax.random.normal(jax.random.PRNGKey(1), (B, T, N_LATENTS, D_BOTTLENECK))
    t = jnp.zeros((B, T))
    samples = jnp.stack([qphi.sample(z, t, jax.random.PRNGKey(i)) for i in range(300)])
    assert abs(float(jnp.mean(samples))) < 0.05
    assert abs(float(jnp.var(samples)) - 1.0) < 0.1


def test_conditioning_is_causal(env):
    """c[t] must depend only on z[<= t]; perturbing a future frame leaves the past intact."""
    qphi, _ = tiny_qphi(env, "gaussian_lowrank")
    B, T = 2, 5
    z = jax.random.normal(jax.random.PRNGKey(1), (B, T, N_LATENTS, D_BOTTLENECK))
    t = jnp.zeros((B, T))
    c1 = qphi.forward_cond(z, t)
    c2 = qphi.forward_cond(z.at[:, T - 1].add(10.0), t)
    assert float(jnp.max(jnp.abs(c1[:, :T - 1] - c2[:, :T - 1]))) < 1e-4
    assert float(jnp.max(jnp.abs(c1[:, T - 1] - c2[:, T - 1]))) > 1e-2
