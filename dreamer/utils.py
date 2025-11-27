import jax
import jax.numpy as jnp
from functools import lru_cache
from dreamer.data import patchify, unpatchify
import orbax.checkpoint as ocp
from pathlib import Path
from flax.core import freeze, unfreeze, FrozenDict
from einops import rearrange
from enum import IntEnum

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

def try_restore(mngr: ocp.CheckpointManager, state_example: dict, meta_example: dict | None = None):
    """
    Build abstract trees from current shapes/dtypes so Orbax can restore safely
    (StandardRestore wants an abstract tree). :contentReference[oaicite:3]{index=3}
    """
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)   # :contentReference[oaicite:4]{index=4}
    restore_args = ocp.args.Composite(
        state=ocp.args.StandardRestore(abstract_state),                                      # :contentReference[oaicite:5]{index=5}
        meta=ocp.args.JsonRestore() if meta_example is not None else None
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



def make_mask(modality_ids:jnp.ndarray, mode: str):
    # Returns a (S,S) boolean mask indicating allowed key for each query index, per mode.
    # S = number of tokens in a single frame.
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
    elif mode in ["wm_agent"]:
        # wm_agent: agent reads all; all non-agent tokens read all *except* agent.
        is_agent_q = (q_mod == Modality.AGENT)
        is_agent_k = (k_mod == Modality.AGENT)
        mask = is_agent_q<=is_agent_k
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Save (1,1,S,S) so it broadcasts over batch*time and heads -> (B*T, 1, S, S)
    modality_mask = mask[None, :, :]                   # (1,S,S)
    modality_mask = jax.lax.stop_gradient(modality_mask)
    return modality_mask

    
