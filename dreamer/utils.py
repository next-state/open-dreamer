import jax
import jax.numpy as jnp
from dreamer.data import patchify, unpatchify
import orbax.checkpoint as ocp
from pathlib import Path
import flax
from flax.core import freeze, unfreeze, FrozenDict
from einops import rearrange, repeat
from enum import IntEnum
from typing import Tuple
import numpy as np
import math

# # --- helpers ---
# temporal_patchify = jax.jit(
#     jax.vmap(patchify, in_axes=(1, None), out_axes=1),  # (B,T,H,W,C) -> (B,T,Np,Dp)
#     static_argnames=("patch",),
# )

# temporal_unpatchify = jax.jit(
#     jax.vmap(unpatchify, in_axes=(1, None, None, None, None), out_axes=1),
#     static_argnames=("H", "W", "C", "patch"),
# )

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

@flax.struct.dataclass  # immutable, PyTree-friendly
class TokenLayout:
    """
    Ordered token layout for a single timestep: segments define the order.
    """
    segments: Tuple[Tuple[Modality, int], ...]  # e.g., ((Modality.LATENT, n_latents), (Modality.IMAGE, n_patches), ...)

    def S(self) -> int:
        return sum(n for _, n in self.segments)

    def modality_ids(self) -> jnp.ndarray:
        parts = [jnp.full((n,), int(m), dtype=jnp.int32) for m, n in self.segments]
        return jnp.concatenate(parts)

    def slices(self) -> dict:
        """Convenience: start/stop indices per modality (first occurrence if repeated)."""
        idx = 0
        out = {}
        for m, n in self.segments:
            out[m] = slice(idx, idx + n)
            idx += n
        return out

    def make_mask(self, mode: str, B: int, T: int):
        """
        Returns a (B*T, 1, S, S) boolean mask indicating allowed key for each query index, per mode.
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
        modality_ids = self.modality_ids()
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
        mask = repeat(mask, "q k -> (b t) h q k", b=B, t=T, h=1)
        mask = jax.lax.stop_gradient(mask)
        return mask


def normalize_with_dataset_stats(videos, *, mean, std):
    """
    Normalize videos using dataset-level statistics.
    
    Handles spatial images (B, T, H, W, C).
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
    
    mean_c = jnp.expand_dims(mean_arr, axis=(0, 1, 2, 3)) 
    std_c =  jnp.expand_dims(std_arr, axis=(0, 1, 2, 3)) 
    
    return (videos - mean_c) / std_c

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
    
    mean_c = jnp.expand_dims(mean_arr, axis=(0, 1, 2, 3)) 
    std_c =  jnp.expand_dims(std_arr, axis=(0, 1, 2, 3)) 
    
    return normalized_videos * std_c + mean_c

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


def make_state(params, opt_state, rng, step):
    # Pack training state as a PyTree; JAX/Orbax-friendly types only.
    return {
        "params": params,
        "opt_state": opt_state,
        "rng": rng,
        "step": jnp.int32(step),
    }

def make_manager(ckpt_dir: str, max_to_keep: int = 5, save_interval_steps: int = 1000, item_names=("state","meta")):
    path = Path(ckpt_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep,
                                           save_interval_steps=save_interval_steps)
    # item_names gives nice attribute access on restore: restored.state, restored.meta
    mngr = ocp.CheckpointManager(path, options=options, item_names=item_names)
    return mngr

def try_restore(mngr: ocp.CheckpointManager, state_example: dict, meta: dict | None = None):
    """
    Build abstract trees from current shapes/dtypes so Orbax can restore safely
    (StandardRestore wants an abstract tree). :contentReference[oaicite:3]{index=3}
    """
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)   # :contentReference[oaicite:4]{index=4}
    restore_args = ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),                                      # :contentReference[oaicite:5]{index=5}
        meta=ocp.args.JsonRestore() if meta is not None else None
    )
    latest = mngr.latest_step()
    if latest is None:
        return None
    restored = mngr.restore(latest, args=restore_args)
    return latest, restored

def maybe_save(mngr: ocp.CheckpointManager, step: int, state: dict, meta: dict | None = None):
    if not mngr.should_save(step):  # obey save interval policy
        return
    save_args = ocp.args.Composite(
        state=ocp.args.StandardSave(state),
        meta=ocp.args.JsonSave(meta) if meta is not None else None
    )
    mngr.save(step, args=save_args)  # async by default; runs in a background thread. :contentReference[oaicite:6]{index=6}




def init_tokenizer(rng, tokenizer, tokenizer_cfg):
    rng, params_rng, mae_rng, dropout_rng, key = jax.random.split(rng, 5)
    dummy = jax.random.uniform(
        key,
        (tokenizer_cfg.dataset.B, tokenizer_cfg.dataset.T, tokenizer_cfg.dataset.H, tokenizer_cfg.dataset.W, tokenizer_cfg.dataset.C),
    )
    variables = tokenizer.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        dummy,
        deterministic=True,
    )
    return rng, variables

def init_dynamics(rng, dynamics, tokenizer_cfg):
    rng, params_rng, dropout_rng, key = jax.random.split(rng, 4)
    n_tokens = tokenizer_cfg.dataset.H * tokenizer_cfg.dataset.W // tokenizer_cfg.patch_size ** 2
    dummy = jax.random.uniform(
        key,
        (tokenizer_cfg.dataset.B, tokenizer_cfg.dataset.T, n_tokens, tokenizer_cfg.encoder.d_bottleneck),
    )
    variables = dynamics.init(
        {"params": params_rng, "dropout": dropout_rng},
        dummy,
        deterministic=True,
    )
    return rng, variables

# -------- Training utilities (shared across scripts) --------

def _ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    """Convert float32 image to uint8."""
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    """Stack images horizontally."""
    return np.concatenate(imgs_hwC, axis=1)

def _tile_videos(trip_list_hwC: list[np.ndarray], *, ncols: int = 2, pad_color: int = 0) -> np.ndarray:
    """Tile a list of videos into a grid."""
    if len(trip_list_hwC) == 0:
        raise ValueError("Empty video list")
    H, W3, C = trip_list_hwC[0].shape
    B = len(trip_list_hwC)
    nrows = math.ceil(B / ncols)
    total = nrows * ncols
    if total > B:
        blank = np.full((H, W3, C), pad_color, dtype=trip_list_hwC[0].dtype)
        trip_list_hwC = trip_list_hwC + [blank] * (total - B)
    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = trip_list_hwC[idx:idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)
    return grid

def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder,
    decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    """Load pretrained tokenizer checkpoint."""
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    import optax
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta
    
def from_dict(cls, d):
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        t = field_types[k]
        if hasattr(t, "__dataclass_fields__"):
            kwargs[k] = from_dict(t, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)
