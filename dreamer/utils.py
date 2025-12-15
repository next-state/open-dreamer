import jax
import jax.numpy as jnp
from dreamer.data import patchify, unpatchify
import orbax.checkpoint as ocp
from pathlib import Path
from flax.core import freeze, unfreeze, FrozenDict
from einops import rearrange
from enum import IntEnum
from omegaconf import OmegaConf, DictConfig
from dataclasses import asdict, is_dataclass

class Modality(IntEnum):
    LATENT   = -1
    IMAGE    = 0
    ACTION   = 1
    PROPRIO  = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7
    # add more as needed


# --- helpers ---
temporal_patchify = jax.jit(
    jax.vmap(patchify, in_axes=(1, None), out_axes=1),  # (B,T,H,W,C) -> (B,T,Np,Dp)
    static_argnames=("patch",),
)

temporal_unpatchify = jax.jit(
    jax.vmap(unpatchify, in_axes=(1, None, None, None, None), out_axes=1),
    static_argnames=("H", "W", "C", "patch"),
)


def normalize_with_dataset_stats(videos, *, mean, std):
    """
    Normalize videos using dataset-level statistics.
    
    Handles both flattened patches (B, T, N, patch*patch*C) and spatial images (B, T, H, W, C).
    For flattened patches, tiles the per-channel stats to match the interleaved layout.
    
    Args:
        videos: input videos/patches
        mean: dataset mean (list of C floats for per-channel)
        std: dataset std (list of C floats for per-channel)
    Returns:
        normalized videos
    """
    mean_arr = jnp.asarray(mean, dtype=videos.dtype)
    std_arr = jnp.asarray(std, dtype=videos.dtype)
    
    last_dim = videos.shape[-1]
    num_channels = len(mean)
    
    if last_dim > num_channels and last_dim % num_channels == 0:
        # flattened patches: tile stats to match interleaved layout
        num_pixels = last_dim // num_channels
        mean_arr = jnp.tile(mean_arr, num_pixels)
        std_arr = jnp.tile(std_arr, num_pixels)
    
    return (videos - mean_arr) / std_arr


def unnormalize_with_dataset_stats(normalized_videos, *, mean, std):
    """
    Unnormalize videos using dataset-level statistics.
    
    Handles both flattened patches (B, T, N, patch*patch*C) and spatial images (B, T, H, W, C).
    For flattened patches, tiles the per-channel stats to match the interleaved layout.
    
    Args:
        normalized_videos: normalized videos/patches
        mean: dataset mean (list of C floats for per-channel)
        std: dataset std (list of C floats for per-channel)
    Returns:
        unnormalized videos
    """
    mean_arr = jnp.asarray(mean, dtype=normalized_videos.dtype)
    std_arr = jnp.asarray(std, dtype=normalized_videos.dtype)
    
    last_dim = normalized_videos.shape[-1]
    num_channels = len(mean)
    
    if last_dim > num_channels and last_dim % num_channels == 0:
        # flattened patches: tile stats to match interleaved layout
        num_pixels = last_dim // num_channels
        mean_arr = jnp.tile(mean_arr, num_pixels)
        std_arr = jnp.tile(std_arr, num_pixels)
    
    return normalized_videos * std_arr + mean_arr


def pack_bottleneck_to_spatial(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
    Requires: N_b == n_spatial * k  (e.g., 512 -> 256 with k=2).
    """
    return rearrange(z_btLd, 'b t (n_spatial k) d -> b t n_spatial (k d)', n_spatial=n_spatial, k=k)


def unpack_spatial_to_bottleneck(z_btLd, *, n_spatial: int, k: int):
    """
    (B,T,S_z, D_z_pre) -> (B,T,N_b,D_b) by splitting D_z_pre into k channels along N_b.
    Requires: N_b == n_spatial * k  (e.g., 256 -> 512 with k=2).
    """
    return rearrange(z_btLd, 'b t n_spatial (k d) -> b t (n_spatial k) d', n_spatial=n_spatial, k=k)


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


def make_mask(modality_ids: jnp.ndarray, mode: str):
    """
    Returns a (S,S) boolean mask indicating allowed key for each query index, per mode.
    S = number of tokens in a single frame.

    Modes:
    - "encoder":
        - Latent tokens (query) can attend to ALL tokens (key).
        - Non-latent tokens (query) can ONLY attend to tokens of the SAME modality (key).
    - "decoder":
        - Latent tokens (query) can ONLY attend to Latent tokens (key).
        - Non-latent tokens (query) can attend to tokens of the SAME modality AND Latent tokens (key).
    - "wm_agent":
        - Action tokens (query) can ONLY attend to Action tokens (key).
        - Observation tokens (query) can attend to Observation AND Action tokens (key).
        - Agent tokens (query) can attend to ALL tokens (key).
    """
    S = int(modality_ids.shape[0])

    # Broadcast helpers
    q_idx = jnp.arange(S)[:, None]       # (S,1)
    k_idx = jnp.arange(S)[None, :]       # (1,S)

    q_mod = modality_ids[q_idx]      # (S,1)
    k_mod = modality_ids[k_idx]      # (1,S)

    if mode == "encoder":
        # latents -> all; non-latents -> same modality only
        mask = (q_mod == k_mod) | (q_mod == Modality.LATENT)
    elif mode == "decoder":
        # latents -> latents only; non-latents -> same modality + latents
        mask = (q_mod == k_mod) | (k_mod == Modality.LATENT)
    elif mode == "wm_agent":
        # wm_agent:

        # Hierarchy levels: Action=0, Obs=1, Agent=2
        # mask = level(q) >= level(k)
        
        def get_level(mod):
            # Default to 1 (Obs)
            lvl = jnp.ones_like(mod, dtype=jnp.int32) # Default to 1 (Obs)
            lvl = jnp.where(mod == Modality.ACTION, 0, lvl) # Set to 0 if Action
            lvl = jnp.where(mod == Modality.AGENT, 2, lvl) # Set to 2 if Agent
            
            return lvl

        q_level = get_level(q_mod)
        k_level = get_level(k_mod)
        
        mask = q_level >= k_level
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Save (S,S)
    modality_mask = jax.lax.stop_gradient(mask)
    return modality_mask


# -------- Model Factory Helpers --------

def create_tokenizer_models(tokenizer_cfg):
    """
    Instantiate Encoder and Decoder from config.
    """
    from dreamer.models import Encoder, Decoder # Local import to avoid circular dependency
    
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


def create_dynamics_model(dynamics_cfg, tokenizer_cfg):
    """
    Instantiate Dynamics model.
    """
    from dreamer.models import Dynamics # Local import
    
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
    from dreamer.models import Dynamics
    
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
