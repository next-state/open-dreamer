import jax
import jax.numpy as jnp
from dreamer.configs import (
    TokenizerModelConfig, 
    DynamicsModelConfig,
    BCRewModelConfig,
    PolicyModelConfig,
)
from dreamer.models import (
    Encoder,
    Decoder,
    Dynamics,
    TaskEmbedder,
    PolicyHeadMTP,
    RewardHeadMTP,
    ValueHead,
)
import orbax.checkpoint as ocp
from pathlib import Path
from flax.core import freeze, unfreeze, FrozenDict
from omegaconf import OmegaConf, DictConfig
from dataclasses import asdict, is_dataclass


# -------- Checkpoint helpers --------
def with_params(variables, new_params):
    # works whether `variables` is a FrozenDict or a plain dict
    d = unfreeze(variables) if isinstance(variables, FrozenDict) else dict(variables)
    d["params"] = new_params
    return freeze(d)


# Pack params so we can optimize both modules with one optimizer.
def pack_mae_params(enc_vars, dec_vars):
    return FrozenDict({
        "enc": enc_vars["params"],
        "dec": dec_vars["params"],
    })


def unpack_mae_params(packed_params, enc_vars, dec_vars):
    enc_vars = with_params(enc_vars, packed_params["enc"])
    dec_vars = with_params(dec_vars, packed_params["dec"])
    return enc_vars, dec_vars


def make_manager(ckpt_dir, max_to_keep=2, save_every=1000, item_names=("state", "meta")):
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, save_interval_steps=save_every)
    return ocp.CheckpointManager(ckpt_dir, options=options, item_names=item_names)


def setup_experiment_checkpointing(cfg: DictConfig | object, run_dir: Path):
    """
    Initializes checkpoint manager and merges checkpoint config over current config.
    
    Strategy:
    1. Start with Saved config (from checkpoint)
    2. Merge Current config (from YAML/CLI) OVER Saved
       -> Allows changing experiment params (LR, max_steps, logging)
    3. Force 'model' and 'dataset' sections BACK to Saved
       -> Ensures architecture and data shapes match the weights
    
    Returns:
        cfg: The merged configuration.
        mngr: The Orbax CheckpointManager.
        start_step: The step to resume from (0 if no checkpoint).
    """
    # 1. Setup Manager
    ckpt_dir = run_dir / "checkpoints"

    ckpt_cfg = getattr(cfg, 'ckpt', None)
    max_to_keep = ckpt_cfg.max_to_keep if ckpt_cfg else 5
    save_every = ckpt_cfg.save_every if ckpt_cfg else 10000
    
    mngr = make_manager(ckpt_dir, max_to_keep, save_every)
    latest_step = mngr.latest_step()

    start_step = 0
    
    # 2. If Checkpoint exists, Load Config & Merge
    if latest_step is not None:
        print(f"[ckpt] Found checkpoint at step {latest_step} in {ckpt_dir}. Loading metadata...")
        
        # Load ONLY metadata
        restored = mngr.restore(latest_step, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
        saved_cfg_dict = restored.meta
        
        if saved_cfg_dict:
            # Saved config (from disk)
            saved_conf = OmegaConf.create(saved_cfg_dict)
            
            # Current config (defaults + CLI overrides)
            current_conf = OmegaConf.create(cfg) 
            
            # Use schema to ensure type safety and proper reconstruction
            cfg_cls = type(cfg)
            schema = OmegaConf.structured(cfg_cls)
            
            # Start with: Schema <- Current (user's config takes precedence)
            merged_conf = OmegaConf.merge(schema, current_conf)
            
            # Override ONLY the model config from checkpoint (architecture must match weights)
            if "model" in saved_conf:
                print("[ckpt] Enforcing model config from checkpoint.")
                merged_conf.model = OmegaConf.merge(schema.model, saved_conf.model)
            
            # Update original cfg object
            cfg = OmegaConf.to_object(merged_conf)
                
            print("[ckpt] Config merged successfully.")
            start_step = latest_step
        else:
            print("[ckpt] Warning: Checkpoint found but no metadata/config present.")

    return cfg, mngr, start_step


def maybe_save_snapshot(mngr, step: int, params, opt_state, rng, cfg: object):
    """
    Saves a full snapshot: Params, Optimizer, RNG, and Configuration.
    """
    if not mngr.should_save(step):  # obey interval policy
        return

    state = {
        "params": params,
        "opt_state": opt_state,
        "rng": rng,
        "step": jnp.int32(step),
    }
    
    meta = asdict(cfg) if is_dataclass(cfg) else OmegaConf.to_container(cfg)

    save_args = ocp.args.Composite(
        state=ocp.args.StandardSave(state),
        meta=ocp.args.JsonSave(meta)
    )
    mngr.save(step, args=save_args)


def load_checkpoint_model_config(ckpt_path: Path | str) -> DictConfig:
    """
    Loads only the 'model' sub-configuration from a checkpoint directory.
    Returns it as an OmegaConf DictConfig.
    """
    path = Path(ckpt_path).expanduser().resolve()
    if (path / "checkpoints").exists():
        path = path / "checkpoints"
    # We only need 'meta'
    mngr = ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(), item_names=("meta",))
    latest = mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    
    restored = mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    
    meta = OmegaConf.create(restored.meta)
    if not "model" in meta:
         raise ValueError(f"Checkpoint at {path} does not contain 'model' config in metadata.")
         
    return meta.model


def load_snapshot_weights(mngr, step, params, opt_state, rng):
    if step == 0:
        return params, opt_state, rng
    """
    Restores the heavy weights into the provided shapes.
    Should be called AFTER models are initialized with the merged config.
    """
    # Create structure based on current initialized arrays
    structure = {
        "params": params,
        "opt_state": opt_state,
        "rng": rng,
        "step": jnp.int32(0)
    }
    abstract_tree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, structure)
    
    restore_args = ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_tree),
        meta=ocp.args.JsonRestore() # Load meta again just to satisfy item_names, or use None
    )
    
    restored = mngr.restore(step, args=restore_args)
    return restored.state["params"], restored.state["opt_state"], restored.state["rng"]


# -------- Model Factory Helpers --------


def create_tokenizer_models(tokenizer_cfg):
    """
    Instantiate Encoder and Decoder from config.
    """
    enc_kwargs = asdict(tokenizer_cfg.encoder)
    dec_kwargs = asdict(tokenizer_cfg.decoder)
    
    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    return encoder, decoder


def init_tokenizer_vars(encoder, decoder, *, input_shape: tuple[int, int, int, int], rng):
    """
    Initialize variables (params + constants) for Encoder and Decoder.
    input_shape: (B, T, Np, D_patch) - Shape of patches.
    """
    rng, enc_rng, dec_rng, mae_rng, drop_rng = jax.random.split(rng, 5)
    
    # Encoder init
    # We need a dummy input to init parameters.
    # input_shape is (B, T, Np, D_patch)
    dummy_patches = jnp.zeros(input_shape, dtype=jnp.float32)
    
    enc_vars = encoder.init(
        {"params": enc_rng, "mae": mae_rng, "dropout": drop_rng},
        dummy_patches, deterministic=True
    )

    # Decoder init
    # Decoder takes bottleneck output: (B, T, n_latents, d_bottleneck)
    n_latents = encoder.n_latents
    d_bottleneck = encoder.d_bottleneck
    B, T = input_shape[0], input_shape[1]
    
    fake_z = jnp.zeros((B, T, n_latents, d_bottleneck), dtype=jnp.float32)
    dec_vars = decoder.init(
        {"params": dec_rng, "dropout": drop_rng},
        fake_z, deterministic=True
    )
    
    return rng, enc_vars, dec_vars


def load_pretrained_tokenizer(tokenizer_ckpt_path: str | Path):
    tokenizer_model_cfg_dict = load_checkpoint_model_config(tokenizer_ckpt_path)
    
    # Cast to TokenizerModelConfig object hierarchy
    raw_cfg = OmegaConf.create(tokenizer_model_cfg_dict)
    schema = OmegaConf.structured(TokenizerModelConfig)
    merged_cfg = OmegaConf.merge(schema, raw_cfg)
    tokenizer_cfg = OmegaConf.to_object(merged_cfg)
    
    encoder, decoder = create_tokenizer_models(tokenizer_cfg)

    rng = jax.random.PRNGKey(0)
    n_patches = tokenizer_cfg.encoder.n_patches 
    d_patch = tokenizer_cfg.decoder.d_patch
    dummy_B, dummy_T = 1, 1
    input_shape = (dummy_B, dummy_T, n_patches, d_patch)
    
    rng, enc_vars, dec_vars = init_tokenizer_vars(encoder, decoder, input_shape=input_shape, rng=rng)

    tokenizer_ckpt_abs = Path(tokenizer_ckpt_path).expanduser().resolve() / "checkpoints"
    mngr = make_manager(tokenizer_ckpt_abs, item_names=("state",))
    
    # Restore full state structure (no abstract params) to avoid mismatch issues
    latest = mngr.latest_step()
    restored = mngr.restore(latest, args=ocp.args.Composite(state=ocp.args.StandardRestore()))

    loaded_params = restored.state["params"]
    enc_vars = with_params(enc_vars, loaded_params["enc"])
    dec_vars = with_params(dec_vars, loaded_params["dec"])

    print(f"[tokenizer] Restored weights from step {latest}")
    return encoder, decoder, enc_vars, dec_vars, tokenizer_cfg


def create_dynamics_model(dynamics_cfg: DynamicsModelConfig, tokenizer_cfg: TokenizerModelConfig):
    """
    Instantiate Dynamics model.
    """
    dynamics_cfg.compute_derived(tokenizer_cfg.encoder)
        
    dyn_kwargs = asdict(dynamics_cfg)
    dyn_kwargs.pop('packing_factor', None) # Helper for validation, not a param
    
    dynamics = Dynamics(**dyn_kwargs)
    return dynamics, dyn_kwargs


def init_dynamics_vars(dynamics, *, 
                       spatial_shape: tuple[int, int, int, int], 
                       k_max: int, 
                       rng):
    """
    Initialize variables for Dynamics model.
    spatial_shape: (B, T, n_spatial, d_spatial) - packed spatial tokens
    """
    rng, key_params, key_drop = jax.random.split(rng, 3)
    
    B, T = spatial_shape[0], spatial_shape[1]
    
    # Dummy inputs for init
    emax = jnp.log2(k_max).astype(jnp.int32)
    actions = jnp.zeros((B, T), dtype=jnp.int32)
    step_idx = jnp.full((B, T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((B, T), k_max - 1, dtype=jnp.int32)
    z1 = jnp.zeros(spatial_shape, dtype=jnp.float32)
    
    dyn_vars = dynamics.init(
        {"params": key_params, "dropout": key_drop}, 
        actions, step_idx, sigma_idx, z1
    )
    
    return dyn_vars


def load_pretrained_dynamics(dynamics_ckpt_path: str | Path):
    """
    Load pretrained dynamics model from checkpoint.
    Returns: dynamics, dyn_vars, dynamics_cfg
    """
    dynamics_model_cfg_dict = load_checkpoint_model_config(dynamics_ckpt_path)
    
    # Cast to DynamicsModelConfig object hierarchy
    raw_cfg = OmegaConf.create(dynamics_model_cfg_dict)
    schema = OmegaConf.structured(DynamicsModelConfig)
    merged_cfg = OmegaConf.merge(schema, raw_cfg)
    dynamics_cfg = OmegaConf.to_object(merged_cfg)
    
    # We need tokenizer config to compute derived, but we'll get it from the checkpoint metadata
    # For now, assume it's already computed in the saved config
    dyn_kwargs = asdict(dynamics_cfg)
    dyn_kwargs.pop('packing_factor', None)  # Helper for validation, not a param
    dynamics = Dynamics(**dyn_kwargs)
    
    # Initialize vars with dummy shapes (will be restored from checkpoint)
    rng = jax.random.PRNGKey(0)
    # Use default shapes - actual shapes will come from checkpoint
    dummy_B, dummy_T = 1, 1
    dummy_n_spatial = dynamics_cfg.n_spatial if dynamics_cfg.n_spatial > 0 else 8
    dummy_d_spatial = dynamics_cfg.d_spatial if dynamics_cfg.d_spatial > 0 else 64
    spatial_shape = (dummy_B, dummy_T, dummy_n_spatial, dummy_d_spatial)
    
    dyn_vars = init_dynamics_vars(
        dynamics,
        spatial_shape=spatial_shape,
        k_max=dynamics_cfg.k_max,
        rng=rng
    )
    
    dynamics_ckpt_abs = Path(dynamics_ckpt_path).expanduser().resolve() / "checkpoints"
    mngr = make_manager(dynamics_ckpt_abs, item_names=("state",))
    
    latest = mngr.latest_step()
    restored = mngr.restore(latest, args=ocp.args.Composite(state=ocp.args.StandardRestore()))
    
    loaded_params = restored.state["params"]
    # Dynamics params might be packed as {"dyn": ...} or just the params directly
    if isinstance(loaded_params, dict) and "dyn" in loaded_params:
        dyn_params = loaded_params["dyn"]
    else:
        dyn_params = loaded_params
    
    dyn_vars = with_params(dyn_vars, dyn_params)
    
    print(f"[dynamics] Restored weights from step {latest}")
    return dynamics, dyn_vars, dynamics_cfg


def create_bc_rew_models(bc_rew_cfg: BCRewModelConfig):
    """
    Instantiate TaskEmbedder, PolicyHeadMTP, and RewardHeadMTP from config.
    """
    task_kwargs = asdict(bc_rew_cfg.task_embedder)
    policy_kwargs = asdict(bc_rew_cfg.policy_head)
    reward_kwargs = asdict(bc_rew_cfg.reward_head)
    
    task_embedder = TaskEmbedder(**task_kwargs)
    policy_head = PolicyHeadMTP(**policy_kwargs)
    reward_head = RewardHeadMTP(**reward_kwargs)
    
    return task_embedder, policy_head, reward_head


def init_bc_rew_vars(
    task_embedder,
    policy_head,
    reward_head,
    *,
    B: int,
    T: int,
    rng
):
    """
    Initialize variables for bc_rew models.
    """
    rng, task_rng, policy_rng, reward_rng = jax.random.split(rng, 4)
    
    # Task embedder init
    dummy_task_ids = jnp.zeros((B,), dtype=jnp.int32)
    task_vars = task_embedder.init({"params": task_rng}, dummy_task_ids, B, T)
    
    # Policy and reward heads init
    dummy_h = jnp.zeros((B, T, task_embedder.d_model), dtype=jnp.float32)
    policy_vars = policy_head.init(
        {"params": policy_rng, "dropout": policy_rng}, 
        dummy_h, 
        deterministic=True
    )
    reward_vars = reward_head.init(
        {"params": reward_rng, "dropout": reward_rng}, 
        dummy_h, 
        deterministic=True
    )
    
    return rng, task_vars, policy_vars, reward_vars


def load_pretrained_bc_rew(bc_rew_ckpt_path: str | Path):
    """
    Load pretrained bc_rew models (dynamics, task_embedder, policy_head, reward_head) from checkpoint.
    Returns: dynamics, task_embedder, policy_head, reward_head, dyn_vars, task_vars, pi_vars, rew_vars, bc_rew_cfg
    """
    bc_rew_model_cfg_dict = load_checkpoint_model_config(bc_rew_ckpt_path)
    
    # Extract dynamics config from raw checkpoint config (before schema merge)
    raw_cfg = OmegaConf.create(bc_rew_model_cfg_dict)
    dynamics_cfg_dict = raw_cfg.get("dynamics")
    if dynamics_cfg_dict is None:
        raise ValueError(f"Checkpoint at {bc_rew_ckpt_path} does not contain 'dynamics' config.")
    
    # Cast dynamics config
    dyn_schema = OmegaConf.structured(DynamicsModelConfig)
    dyn_merged = OmegaConf.merge(dyn_schema, dynamics_cfg_dict)
    dynamics_cfg = OmegaConf.to_object(dyn_merged)
    
    # Cast to BCRewModelConfig object hierarchy (without dynamics)
    schema = OmegaConf.structured(BCRewModelConfig)
    merged_cfg = OmegaConf.merge(schema, raw_cfg)
    bc_rew_cfg = OmegaConf.to_object(merged_cfg)
    
    # Compute derived values from dynamics config
    bc_rew_cfg.compute_derived(dynamics_cfg)
    
    # Create models
    dyn_kwargs = asdict(dynamics_cfg)
    dyn_kwargs.pop('packing_factor', None)
    dynamics = Dynamics(**dyn_kwargs)
    
    task_embedder, policy_head, reward_head = create_bc_rew_models(bc_rew_cfg)
    
    # Initialize vars with dummy shapes
    rng = jax.random.PRNGKey(0)
    dummy_B, dummy_T = 1, 1
    dummy_n_spatial = dynamics_cfg.n_spatial if dynamics_cfg.n_spatial > 0 else 8
    dummy_d_spatial = dynamics_cfg.d_spatial if dynamics_cfg.d_spatial > 0 else 64
    spatial_shape = (dummy_B, dummy_T, dummy_n_spatial, dummy_d_spatial)
    
    dyn_vars = init_dynamics_vars(
        dynamics,
        spatial_shape=spatial_shape,
        k_max=dynamics_cfg.k_max,
        rng=rng
    )
    
    rng, task_vars, pi_vars, rew_vars = init_bc_rew_vars(
        task_embedder, policy_head, reward_head,
        B=dummy_B, T=dummy_T, rng=rng
    )
    
    # Load from checkpoint
    bc_rew_ckpt_abs = Path(bc_rew_ckpt_path).expanduser().resolve() / "checkpoints"
    mngr = make_manager(bc_rew_ckpt_abs, item_names=("state",))
    
    latest = mngr.latest_step()
    restored = mngr.restore(latest, args=ocp.args.Composite(state=ocp.args.StandardRestore()))
    
    loaded_params = restored.state["params"]
    # BC/rew params are stored as {"dyn": ..., "task": ..., "pi": ..., "rew": ...}
    dyn_params = loaded_params.get("dyn", loaded_params) if isinstance(loaded_params, dict) else loaded_params
    task_params = loaded_params.get("task") if isinstance(loaded_params, dict) else None
    pi_params = loaded_params.get("pi") if isinstance(loaded_params, dict) else None
    rew_params = loaded_params.get("rew") if isinstance(loaded_params, dict) else None
    
    dyn_vars = with_params(dyn_vars, dyn_params)
    if task_params is not None:
        task_vars = with_params(task_vars, task_params)
    if pi_params is not None:
        pi_vars = with_params(pi_vars, pi_params)
    if rew_params is not None:
        rew_vars = with_params(rew_vars, rew_params)
    
    print(f"[bc_rew] Restored weights from step {latest}")
    # Add dynamics config to bc_rew_cfg for access by callers
    bc_rew_cfg.dynamics = dynamics_cfg
    return dynamics, task_embedder, policy_head, reward_head, dyn_vars, task_vars, pi_vars, rew_vars, bc_rew_cfg


def create_policy_models(policy_cfg: PolicyModelConfig):
    """
    Instantiate PolicyHeadMTP and ValueHead for policy training.
    """
    policy_kwargs = asdict(policy_cfg.policy_head)
    value_kwargs = asdict(policy_cfg.value_head)
    
    policy_head = PolicyHeadMTP(**policy_kwargs)
    value_head = ValueHead(**value_kwargs)
    
    return policy_head, value_head


def init_policy_vars(
    policy_head,
    value_head,
    *,
    B: int,
    T: int,
    rng
):
    """
    Initialize variables for policy models.
    """
    rng, policy_rng, value_rng = jax.random.split(rng, 3)
    
    dummy_h = jnp.zeros((B, T, policy_head.d_model), dtype=jnp.float32)
    policy_vars = policy_head.init(
        {"params": policy_rng, "dropout": policy_rng}, 
        dummy_h, 
        deterministic=True
    )
    value_vars = value_head.init(
        {"params": value_rng, "dropout": value_rng}, 
        dummy_h, 
        deterministic=True
    )
    
    return rng, policy_vars, value_vars
