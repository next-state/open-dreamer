"""
Phase 4: RL policy training
JIT-friendly policy/value training using imagination rollouts.

High-level outline (from the docstring plan):

- Trajectory Generation
    - Roll out the policy in latent space starting from a ground-truth context.
    - Unroll π(a|s) from s0 for `horizon` steps, creating latent states s1…sT.
    - Collect policy actions a1…aT and hidden states h0…hT.

- Reward / Value annotation
    - Use the reward head on h1…hT to get r1…rT.
    - Use the value head on h0…hT to get V0…VT.
    - Compute TD-λ returns G0…G{T-1} using V1…VT and r1…rT (with bootstrap VT).

- Value / Policy updates
    - Train V_head on (s0…s{T-1}) to regress G0…G{T-1}.
    - Train policy head on (s0…s{T-1}, a1…aT, G0…G{T-1}, V0…V{T-1}) using PMPO.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from flax.typing import VariableDict
import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.core import FrozenDict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import orbax.checkpoint as ocp

from dreamer.configs import BCRewConfig, RLConfig
from dreamer.data import make_iterator
from dreamer.logging import MetricLogger
from dreamer.models import Dynamics, PolicyHeadMTP, RewardHeadMTP, TaskEmbedder, Tokenizer, ValueHead
from dreamer.training import (
    compute_policy_loss,
    compute_reward_loss,
    run_evaluation,
    run_agent_visualization,
    shortcut_forcing_step,
)
from dreamer.utils import (
    _ensure_dir,
    make_manager,
    make_state,
    maybe_save,
    try_restore,
    count_parameters_by_component,
    get_lr_schedule,
    from_dict,
    recursive_list_to_tuple
)

# Suppress absl info logs
logging.getLogger('absl').setLevel(logging.WARNING)

# disable preallocation completely
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

@dataclass(frozen=True)
class OptimizerContainer:
    """Hashable container for optimizers to pass as static argument to JIT."""
    policy: optax.GradientTransformationExtraArgs
    value: optax.GradientTransformationExtraArgs


def train_step(
    frames: jnp.ndarray,
    actions: jnp.ndarray,
    rng: jnp.ndarray,
):
    """
    Single training step:
      - encode frames to latents
      - run JIT-friendly imagination rollouts in latent space
      - compute TD-λ value targets and PMPO policy loss
      - update policy/value head params.
    """
    pass




# ---------------------------
# Main
# ---------------------------

def run(cfg: RLConfig):
    """Main training loop for agent finetuning."""
    # Setup directories
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    vis_dir = _ensure_dir(run_dir / "viz")
    print(f"[setup] output dir: {run_dir.resolve()}")
    
    # Wandb
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project or cfg.run_name,
            name=cfg.run_name,
            config=asdict(cfg),
            dir=str(run_dir),
        )
    
    # Load pretrained models from bc_rew checkpoint
    rng = jax.random.PRNGKey(0)
    print(f"[setup] Loading pretrained models from {cfg.bc_rew_ckpt}")
    
    # 1. Restore metadata from bc_rew checkpoint to get config
    bc_rew_mngr = make_manager(cfg.bc_rew_ckpt, item_names=("state", "meta"))
    latest_bc_rew = bc_rew_mngr.latest_step()
    if latest_bc_rew is None:
        raise ValueError(f"No checkpoint found in {cfg.bc_rew_ckpt}")
    
    restored_meta = bc_rew_mngr.restore(latest_bc_rew, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    bc_rew_cfg_dict = restored_meta.meta["cfg"]
    bc_rew_cfg_dict = recursive_list_to_tuple(bc_rew_cfg_dict)
    bc_rew_cfg = from_dict(BCRewConfig, bc_rew_cfg_dict)
    
    print(f"[setup] Loaded bc_rew config from step {latest_bc_rew}")
    print(f"[setup] Loading pretrained dynamics from {bc_rew_cfg.dynamics_ckpt}")
    print(f"[setup] Loading pretrained tokenizer from {bc_rew_cfg.tokenizer_ckpt}")
    
    # 2. Load tokenizer and dynamics (from dynamics checkpoint)
    dynamics, dynamics_vars, dynamics_cfg, tokenizer, tokenizer_vars, tokenizer_cfg = Dynamics.from_pretrained(bc_rew_cfg.dynamics_ckpt)
    dynamics_params = dynamics_vars["params"]
    dynamics_constants = dynamics_vars.get("constants", FrozenDict())
    
    # 3. Initialize task embedder, policy head, and reward head models
    print("[setup] Initializing agent components")
    task_embedder = TaskEmbedder(
        d_model=dynamics.config.d_model,
        n_agent=bc_rew_cfg.n_agent,
        use_ids=bc_rew_cfg.use_task_ids,
        n_tasks=bc_rew_cfg.n_tasks
    )
    bc_policy_head = PolicyHeadMTP(
        d_model=dynamics.config.d_model,
        action_dim=dynamics.config.action_dim,
        L=bc_rew_cfg.L
    )
    reward_head = RewardHeadMTP(
        d_model=dynamics.config.d_model,
        L=bc_rew_cfg.L,
        num_bins=bc_rew_cfg.num_reward_bins,
        log_low=bc_rew_cfg.reward_log_low,
        log_high=bc_rew_cfg.reward_log_high
    )
    
    # 4. Initialize parameters (to get structure for checkpoint restoration)
    rng, task_key, bc_key, rew_key = jax.random.split(rng, 4)
    
    # Dummy inputs for initialization
    dummy_h = jnp.zeros((1, 4, bc_rew_cfg.L, dynamics.config.d_model))  # (B, T, D) for heads
    dummy_task = jnp.zeros((1,), dtype=jnp.int32) if bc_rew_cfg.use_task_ids else jnp.zeros((1, bc_rew_cfg.n_tasks))
    task_embedder_params = task_embedder.init(task_key, task=dummy_task, B=1, T=4)["params"]
    bc_policy_params = bc_policy_head.init(bc_key, dummy_h, deterministic=True)["params"]
    reward_params = reward_head.init(rew_key, dummy_h, deterministic=True)["params"]
    
    # 5. Restore parameters from bc_rew checkpoint
    # Create dummy optimizer states for checkpoint restoration (we only need shapes)
    dummy_optimizer = optax.adamw(1e-4)
    opt_states = {
        "task_embedder": dummy_optimizer.init(task_embedder_params),
        "policy": dummy_optimizer.init(bc_policy_params),
        "reward": dummy_optimizer.init(reward_params),
        "dynamics": dummy_optimizer.init(dynamics_params),
    }
    state_example = make_state(
        {
            "task_embedder": task_embedder_params,
            "policy": bc_policy_params,
            "reward": reward_params,
            "dynamics": dynamics_params,
        },
        {
            "task_embedder": opt_states["task_embedder"],
            "policy": opt_states["policy"],
            "reward": opt_states["reward"],
            "dynamics": opt_states["dynamics"],
        },
        rng,
        step=0
    )
    del opt_states
    
    restored = try_restore(bc_rew_mngr, state_example, meta=None)
    if restored is None:
        raise ValueError(f"Failed to restore checkpoint from {cfg.bc_rew_ckpt}")
    
    latest_step, r = restored
    loaded_params = r.state["params"]
    
    # Extract loaded parameters
    task_embedder_params = loaded_params["task_embedder"]
    bc_policy_params = loaded_params["policy"]
    reward_params = loaded_params["reward"]
    dynamics_params = loaded_params["dynamics"]
    
    print(f"[setup] Restored all models from bc_rew checkpoint (step {latest_step})")

    # Initialize new policy head and value head
    rng, pol_key, val_key = jax.random.split(rng, 3)
    policy_head = PolicyHeadMTP(
        d_model=dynamics.config.d_model,
        action_dim=dynamics.config.action_dim,
        L=1
    )
    value_head = ValueHead(
        d_model=dynamics.config.d_model,
        num_bins=cfg.num_value_bins,
    )
    policy_vars = policy_head.init(pol_key, dummy_h, deterministic=True)
    value_vars = value_head.init(val_key, dummy_h, deterministic=True)
    # Optimizers 
    adamw = partial(optax.adamw, b1=0.9, b2=0.9, weight_decay=1e-4)
    optimizers = OptimizerContainer(
        policy=adamw(cfg.lr_policy),
        value=adamw(cfg.lr_value),
    )
    opt_states = {
        "policy": optimizers.policy.init(policy_vars["params"]),
        "value": optimizers.value.init(value_vars["params"]),
    }

    # Logging & checkpointing
    logger = MetricLogger(
        use_wandb=cfg.use_wandb,
        log_every=cfg.log_every,
        max_steps=cfg.max_steps,
        wandb_obj=wandb,
    )
    mngr = make_manager(ckpt_dir, max_to_keep=cfg.ckpt_max_to_keep, save_interval_steps=cfg.ckpt_save_every)
    # Try to restore checkpoint
    state_example = make_state(
        {
            "policy": policy_vars["params"],
            "value": value_vars["params"],
        },
        {
            "policy": opt_states["policy"],
            "value": opt_states["value"],
        },
        rng,
        step=0
    )
    meta = {"cfg": asdict(cfg)}
    restored = try_restore(mngr, state_example, meta)
    start_step = 0
    if restored is not None:
        latest_step, r = restored
        params = r.state["params"]
        opt_states = r.state["opt_state"]
        rng = r.state["rng"]
        start_step = int(r.state["step"])
        print(f"[ckpt] Restored step {latest_step}")
    else:
        params = {
            "policy": policy_vars["params"],
            "value": value_vars["params"],
        }
    param_counts = count_parameters_by_component(params)
    print(f"Parameter counts: {param_counts}")
    import ipdb; ipdb.set_trace()
    tokenizer_cfg.dataset.p_include_reward = 0
    dataset = make_iterator(tokenizer_cfg.dataset)

    # Training loop
    pbar = tqdm(enumerate(dataset, start=start_step), total=cfg.max_steps)
    for step, batch in pbar:
        if step >= cfg.max_steps:
            break
        
        rng, step_key = jax.random.split(rng, 2)
        
        # Get videos and compute batch size
        videos = batch["videos"]
        actions = batch["actions"]

    


@hydra.main(version_base=None, config_path="../configs", config_name="policy")
def main(cfg: DictConfig):
    schema = OmegaConf.structured(RLConfig)
    cfg = OmegaConf.merge(schema, cfg)
    agent_cfg = OmegaConf.to_object(cfg)
    run(agent_cfg)


if __name__ == "__main__":
    main()