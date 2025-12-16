import jax
import jax.numpy as jnp
from dreamer.data import patchify, unpatchify
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
