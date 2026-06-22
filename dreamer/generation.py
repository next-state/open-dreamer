from dataclasses import dataclass
import einops
import jax
import jax.numpy as jnp
from typing import Any, Tuple
from .models import KVCachesDict, Dynamics, PolicyHeadMTP, Tokenizer
from .actions import Actions
from .utils import normalize_latents, unnormalize_latents
from tqdm import tqdm


@dataclass
class DenoiseSchedule:
    """
    Precomputed, JAX-friendly schedule for the τ-ladder (continuous signal levels).

    DuMo conditions only on the continuous signal level σ (no shortcut step-size embedding), so
    the schedule is just a ladder of σ values plus the Euler/DDIM mixing coefficients. The same
    ladder drives both samplers; only the head differs:
      * head="v" with many steps  -> high-quality multi-step velocity sampling
      * head="u" with few steps    -> DuMo few-step flow-map sampling

    Attributes:
        num_steps: number of sampling steps taken during inference.
        tau_values: signal levels used during denoising, τ = [0, d, 2d, ..., 1 - d, 1].
        beta_values: x-prediction Euler/DDIM mixing coefficients, beta[s] = (1 - τ[s+1]) / (1 - τ[s]).
        tau_ctx: (continuous) noise level applied to context frames during autoregressive rollout.
    """

    num_steps: int
    tau_values: jax.Array
    beta_values: jax.Array
    tau_ctx: jax.Array

    @classmethod
    def init(cls, num_steps: int, tau_ctx_target: float = 0.9, dtype=jnp.float32) -> "DenoiseSchedule":
        """
        Create a DenoiseSchedule object.
        Args:
            num_steps: Number of denoising steps in the schedule.
            tau_ctx_target: Noise level for context frames during autoregressive rollout (σ; 1 = clean).
            dtype: Dtype for precomputed arrays (should match model dtype).
        Returns:
            DenoiseSchedule object.
        """
        tau_values = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=dtype)
        beta_values = (1.0 - tau_values[1:]) / jnp.maximum(1.0 - tau_values[:-1], 1e-8)
        tau_ctx = jnp.array(tau_ctx_target, dtype=dtype)

        return cls(num_steps, tau_values, beta_values, tau_ctx)


# ---------------------------
# Single-step τ-ladder denoiser
# ---------------------------

def next_latent(
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,                   # (B, 1, n_spatial, D_s)
    rng: jax.Array,
    prefill_length: int | None = None,
    task_embedding: jax.Array | None = None,  # (B, T_ctx+1, n_agent, d_model)
    caches: KVCachesDict | None = None,
    latents_ctx: jax.Array| None = None,                     # (B, T_ctx, n_spatial, D_s)
    actions_ctx: Actions | None = None,
    head: str = "v",
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, jax.Array, dict]:
    """
    JAX-friendly τ-ladder denoiser for a single future latent with KV caching.

    Each step predicts the clean latent with the requested DuMo head (x-prediction) and applies
    an x-prediction Euler/DDIM mixing step toward it. With head="v" and many num_steps this is
    the multi-step velocity sampler; with head="u" and small num_steps it is DuMo's few-step
    flow-map sampler.

    Args:
        dynamics: Dynamics NNX model
        schedule: Precomputed DenoiseSchedule
        action: action for current step
        latent_shape: Tuple representing the shape of the latent (B, 1, n_spatial, D_s)
        prefill_length: Number of ground truth latents passed during prefill.
        rng: Random number generator key
        task_embedding: Optional agent tokens (B, T_ctx+1, n_agent, d_model)
        caches: KV cache for context frames (from previous finalized frames)
        latents_ctx: Optional context latents (B, T_ctx, n_spatial, D_s) used for debugging/non-cached mode
        actions_ctx: Optional context actions (B, T_ctx) used for debugging/non-cached mode
        head: DuMo head to denoise with ("v" velocity / multi-step, "u" flow-map / few-step)

    Returns:
        Tuple containing:
        - latent_t_final: The denoised latent (B, 1, n_spatial, D_s)
        - h_last: The final hidden state from dynamics (B, n_agent, d_model)
        - caches_new: The updated KV cache
    """
    rng, rng_latent, rng_ctx = jax.random.split(rng, 3)
    noisy_latent = jax.random.normal(rng_latent, latent_shape)
    B = latent_shape[0]

    latents_ctx_noised = None
    if latents_ctx is not None and caches is None:
        latents_prefill = latents_ctx[:, :prefill_length]
        latents_decode = latents_ctx[:, prefill_length:]
        noise_decode = jax.random.normal(rng_ctx, latents_decode.shape)
        latents_decode_noised = latents_decode * schedule.tau_ctx + (1 - schedule.tau_ctx) * noise_decode
        latents_ctx_noised = jnp.concatenate([latents_prefill, latents_decode_noised], axis=1)

    action = action[:, None, ...]  # expand squeezed-out time dimension

    def refinement_step(latent_t, s):
        beta = schedule.beta_values[s]
        tau_val = schedule.tau_values[s]

        if caches is not None:
            latent_input, actions_input = latent_t, action

            sigma = jnp.full((B, 1), tau_val, dtype=jnp.float32)

            assert task_embedding is None or task_embedding.shape[1] == noisy_latent.shape[1], f"task_embedding.shape = {task_embedding.shape}, noisy_latent.shape = {noisy_latent.shape}"

        else:
            # Used only for debugging.
            assert latents_ctx_noised is not None and actions_ctx is not None and prefill_length is not None
            latent_input  = jnp.concatenate([latents_ctx_noised, latent_t], axis=1)  # (B, T_ctx+1, n_spatial, D_s)
            actions_input = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), actions_ctx, action)  # (B, T_ctx+1)

            decode_length = latents_ctx_noised.shape[1] - prefill_length
            sigma_prefill = jnp.ones((B, prefill_length), dtype=jnp.float32)                          # clean context
            sigma_decode  = jnp.full((B, decode_length), schedule.tau_ctx, dtype=jnp.float32)         # slightly noised context
            sigma_curr    = jnp.full((B, 1), tau_val, dtype=jnp.float32)
            sigma         = jnp.concatenate([sigma_prefill, sigma_decode, sigma_curr], axis=1)        # (B, T_ctx+1)

        # Dynamics call
        latent_clean_pred_seq, (h_seq, _) = dynamics(
            actions_input, sigma, latent_input,
            head=head, task_embeddings=task_embedding, deterministic=True, caches=caches
        )

        latent_clean_pred = latent_clean_pred_seq[:, -1:, :, :]  # (B, 1, n_spatial, D_s)
        h_last = h_seq[:, -1:, :, :] if isinstance(h_seq, jax.Array) else h_seq  # (B, n_agent, d_model)

        # Per-step ODE diagnostics
        x0_norm = jnp.mean(jnp.abs(latent_clean_pred))
        update_mag = jnp.mean(jnp.abs((1.0 - beta) * (latent_clean_pred - latent_t)))

        # Per-step mixing toward clean latent
        latent_t_new = beta * latent_t + (1.0 - beta) * latent_clean_pred

        return latent_t_new, (h_last, x0_norm, update_mag)

    # Run τ-ladder with JAX control flow using scan to keep carry/output structure consistent.
    latent_t_final, (h_history, x0_norm_hist, update_mag_hist) = jax.lax.scan(
        refinement_step,
        noisy_latent,
        jnp.arange(schedule.num_steps),
    )

    # ODE diagnostics
    diag_dict = {
        'x0_norm': x0_norm_hist,
        'update_mag': update_mag_hist,
    }

    if caches is not None:
        # Update caches by doing one more forward pass with tau_ctx
        sigma = jnp.full((B, 1), schedule.tau_ctx, dtype=jnp.float32)

        rng, new_random_key = jax.random.split(rng)
        latent_noised_caching = latent_t_final * schedule.tau_ctx + (1 - schedule.tau_ctx) * jax.random.normal(new_random_key, shape=latent_t_final.shape, dtype=latent_t_final.dtype)

        # Dynamics call
        _, (h_seq_final, caches_new) = dynamics(
            action, sigma, latent_noised_caching,
            head=head, task_embeddings=task_embedding, deterministic=True, caches=caches
        )
        h_last = h_seq_final[:, -1:, :, :] if isinstance(h_seq_final, jax.Array) else h_seq_final
    else:
        h_last = h_history[-1] if h_history is not None else None  # (B, n_agent, d_model)
        caches_new = None

    assert isinstance(h_last, jax.Array) or h_last is None

    latent_t_final = unnormalize_latents(latent_t_final, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    return latent_t_final, h_last, caches_new, rng, diag_dict

def next_frame(
    tokenizer: Tokenizer,
    dynamics: Dynamics,
    schedule: DenoiseSchedule,
    action: Actions,
    latent_shape: Tuple,
    dynamics_cache: Any,
    tokenizer_cache: Any,
    rng: jax.Array,
    task_embedding: jax.Array | None = None,
    head: str = "v",
) -> Tuple[jax.Array, jax.Array | None, KVCachesDict | None, Any, jax.Array]:
    """
    Generate next frame using dynamics model and decode to pixels.

    Args:
        tokenizer: Tokenizer NNX model for decoding
        dynamics: Dynamics NNX model
        schedule: Denoising schedule
        action: Single action to condition on where the time dimension is squeezed (B, ...)
        latent_shape: Shape of latent (B, 1, n_spatial, D_s)
        dynamics_cache: KV cache for dynamics model from previous steps
        tokenizer_cache: KV cache for tokenizer decoder from previous steps
        rng: Random key
        task_embedding: Optional task embedding (currently unused)
        head: DuMo head to denoise with ("v" or "u")

    Returns:
        Tuple of (frame as jax.Array, h_last, updated dynamics cache, updated tokenizer cache, updated rng)
    """
    # Generate next latent using τ-ladder denoising
    latent, h_last, dynamics_cache_updated, rng, _ = next_latent(
        dynamics=dynamics,
        schedule=schedule,
        action=action,
        latent_shape=latent_shape,
        rng=rng,
        prefill_length=None,  # No prefill for interactive generation
        task_embedding=task_embedding,
        caches=dynamics_cache,
        head=head,
    )
    
    # Decoder call
    frame, tokenizer_cache_updated = tokenizer.decode(
        latent,
        caches=tokenizer_cache,
        deterministic=True,
    )
    
    # Clip to valid range (keep as JAX array)
    # frame shape: (B, 1, H, W, C)
    frame = jnp.clip(frame, 0, 255).astype(jnp.uint8)
    
    return frame, h_last, dynamics_cache_updated, tokenizer_cache_updated, rng


def latent_rollout(
    dynamics: Dynamics,
    policy: PolicyHeadMTP | Actions,
    schedule: DenoiseSchedule,
    latents_ctx: jax.Array,
    actions_ctx: Actions,
    num_steps: int,
    rng: jax.Array,
    initial_task_embedding: jax.Array | None = None,
    deterministic: bool = False,
    use_kv_cache: bool = True,
    head: str = "v",
):
    """
    Autoregressive rollout in latent space.

    Args:
        dynamics: Dynamics NNX model.
        policy: Policy NNX model or sequence of actions.
        schedule: DenoiseSchedule.
        latents_ctx: (B, T_ctx, n_spatial, D_s) Context latents.
        actions_ctx: Context actions.
        num_steps: Number of steps to unroll.
        rng: Random number generator key
        initial_task_embedding: Optional (B, T_ctx, n_agent, D) agent tokens for context.
        deterministic: Whether to sample deterministic actions from the policy.
        use_kv_cache: Whether to use KV caching in dynamics for faster rollout.
        head: DuMo head to denoise with ("v" velocity / multi-step, "u" flow-map / few-step).

    Returns:
        Dict with 'latents', 'actions', 'hidden_states', 'context_hidden'
    """
    B, T_ctx, n_spatial, D_s = latents_ctx.shape
    latent_shape = (B, 1, n_spatial, D_s)

    # Normalize context latents for dynamics (keep original for output)
    latents_ctx_orig = latents_ctx
    latents_ctx = normalize_latents(latents_ctx, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)

    # Scan loop for rollout
    def scan_step(carry, step_idx):
        h_t, caches_t, rng = carry

        # Sample action
        rng, rng_policy = jax.random.split(rng)
        
        if isinstance(policy, Actions):
            action = policy[:, step_idx]  # (B, ...)
        else:
            all_actions = policy.sample(h_t, deterministic=deterministic, rng=rng_policy)  # (B, T, L, ...)
            action = all_actions[:, 0, 0, ...]  # (B, ...) - use first predicted action
        
        # Predict next latent (denoising)
        latent_next, h_next, caches_next, rng, diag = next_latent(
            dynamics, schedule, action, latent_shape, rng, caches=caches_t, task_embedding=task_embedding, head=head
        )
        
        return (h_next, caches_next, rng), (latent_next[:, 0], action, h_next, diag) # latent_next[:, 0] to remove time dimension


    if use_kv_cache:
        # Initialize caches and process context
        window_size = min(T_ctx + num_steps, dynamics.cfg.context_length)
        n_agents = policy.cfg.L if isinstance(policy, PolicyHeadMTP) else 0
        caches = dynamics.create_static_caches(batch_size=B, n_latents=n_spatial, window_size=window_size, n_agent=n_agents, dtype=latents_ctx.dtype)

        # Run dynamics on context to prefill caches and get last hidden state
        # Use clean signal (sigma=1.0) for ground truth context
        sigma_prefill = jnp.ones((B, T_ctx), dtype=jnp.float32)

        _, (h_seq, caches) = dynamics(
            actions_ctx, sigma_prefill, latents_ctx,
            head=head, task_embeddings=initial_task_embedding, caches=caches, deterministic=True
        )

        # h_seq: (B, T_ctx, n_agent, D). We need the state at the last context step.
        task_embedding = initial_task_embedding[:, -1:] if isinstance(initial_task_embedding, jax.Array) else None
        h_last = h_seq[:, -1:] if isinstance(h_seq, jax.Array) else None  # (B, 1, n_agent, D)

        # Run scan
        _, (rollout_latents, rollout_actions, rollout_hidden, rollout_diags) = jax.lax.scan(
            scan_step,
            (h_last, caches, rng),
            jnp.arange(num_steps)
        )

        # Unpack results
        rollout_latents = einops.rearrange(rollout_latents, 't b s d -> b t s d')
        rollout_actions = jax.tree.map(lambda x: einops.rearrange(x, 't b ... -> b t ...'), rollout_actions)
        # h_next has shape (B, 1, n_agent, d_model), so scan output is (t, B, 1, n_agent, d_model)
        rollout_hidden = einops.rearrange(rollout_hidden, 't b 1 n d -> b t n d') if isinstance(rollout_hidden, jax.Array) else None

        # Leaves have shape (num_AR_steps, num_tau_steps) -> (num_tau_steps,)
        ode_diags = jax.tree.map(lambda x: jnp.mean(x, axis=0), rollout_diags)

        out_latents = jnp.concatenate((latents_ctx_orig, rollout_latents), axis=1)
    else:
        # Run scan without KV caching in Python for-loop to support debugging
        
        # Not Implemented:
        h_seq, rollout_hidden = None, None
        if not isinstance(policy, Actions):
                raise NotImplementedError

        pred_latents = []
        for step_idx in tqdm(range(num_steps)):
            action = policy[:, step_idx]

            # Predict next latent
            latent_next, _, _, rng, diag = next_latent(
                dynamics, schedule, action, latent_shape, rng, caches=None, task_embedding=None,
                latents_ctx=latents_ctx, actions_ctx=actions_ctx, prefill_length=T_ctx
            )

            pred_latents.append(latent_next[:, 0])  # Remove time dimension
            latent_next_norm = normalize_latents(latent_next, dynamics.cfg.latent_mean, dynamics.cfg.latent_std)
            latents_ctx = jnp.concatenate([latents_ctx, latent_next_norm], axis=1)
            action = action[:, None, ...]
            actions_ctx = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), actions_ctx, action)

        out_latents = jnp.concatenate((latents_ctx_orig, jnp.stack(pred_latents, axis=1)), axis=1)
        rollout_actions = actions_ctx
        ode_diags = None

    return {
        'latents': out_latents,
        'actions': rollout_actions,
        'hidden_states': rollout_hidden,
        'context_hidden': h_seq,
        'ode_diags': ode_diags,
    }


