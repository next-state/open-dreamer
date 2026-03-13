"""FVD (Fréchet Video Distance) computation using I3D features.

Self-contained port of the JAX FVD implementation. Includes:
- I3D model (pure functional, stateless)
- Weight loading from .npz (with optional PyTorch conversion)
- Preprocessing (resize to 224x224, scale to [-1, 1])
- Batched I3D feature extraction
- Fréchet distance computation

Input convention: videos as (B, T, H, W, C) uint8 [0, 255].
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


# ---------------------------------------------------------------------------
# I3D building blocks (pure functional, NDHWC layout)
# ---------------------------------------------------------------------------

def _unit3d(params, x, stride=(1, 1, 1), use_bn=True, use_relu=True):
    """Conv3D (SAME padding) + optional BatchNorm (inference mode) + optional ReLU."""
    x = lax.conv_general_dilated(
        x, params['conv_w'],
        window_strides=stride,
        padding='SAME',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    if 'conv_b' in params:
        x = x + params['conv_b']
    if use_bn:
        x = ((x - params['bn_mean'])
             / jnp.sqrt(params['bn_var'] + 1e-5)
             * params['bn_scale']
             + params['bn_offset'])
    if use_relu:
        x = jax.nn.relu(x)
    return x


def _maxpool3d(x, window, stride):
    """Max-pool with SAME padding. window/stride are 3-tuples (D, H, W)."""
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1,) + window + (1,),
        window_strides=(1,) + stride + (1,),
        padding='SAME',
    )


def _avgpool3d(x, window, stride):
    """Average-pool with VALID padding. window/stride are 3-tuples (D, H, W)."""
    sum_pool = lax.reduce_window(
        x, 0.0, lax.add,
        window_dimensions=(1,) + window + (1,),
        window_strides=(1,) + stride + (1,),
        padding='VALID',
    )
    return sum_pool / (window[0] * window[1] * window[2])


def _inception_module(params, x):
    """One Inception block with 4 branches, concatenated along channel axis."""
    b0 = _unit3d(params['b0'], x)
    b1 = _unit3d(params['b1b'], _unit3d(params['b1a'], x))
    b2 = _unit3d(params['b2b'], _unit3d(params['b2a'], x))
    b3 = _unit3d(params['b3b'], _maxpool3d(x, (3, 3, 3), (1, 1, 1)))
    return jnp.concatenate([b0, b1, b2, b3], axis=-1)


def _i3d_forward(params, x):
    """Full I3D forward pass.

    Args:
        params: nested dict of arrays from load_i3d_pretrained().
        x: input tensor [B, T, H, W, C] in [-1, 1].

    Returns:
        Logits of shape [B, 400].
    """
    # Stem
    x = _unit3d(params['Conv3d_1a_7x7'], x, stride=(2, 2, 2))
    x = _maxpool3d(x, (1, 3, 3), (1, 2, 2))
    x = _unit3d(params['Conv3d_2b_1x1'], x)
    x = _unit3d(params['Conv3d_2c_3x3'], x)
    x = _maxpool3d(x, (1, 3, 3), (1, 2, 2))

    # Inception blocks 3
    x = _inception_module(params['Mixed_3b'], x)
    x = _inception_module(params['Mixed_3c'], x)
    x = _maxpool3d(x, (3, 3, 3), (2, 2, 2))

    # Inception blocks 4
    x = _inception_module(params['Mixed_4b'], x)
    x = _inception_module(params['Mixed_4c'], x)
    x = _inception_module(params['Mixed_4d'], x)
    x = _inception_module(params['Mixed_4e'], x)
    x = _inception_module(params['Mixed_4f'], x)
    x = _maxpool3d(x, (2, 2, 2), (2, 2, 2))

    # Inception blocks 5
    x = _inception_module(params['Mixed_5b'], x)
    x = _inception_module(params['Mixed_5c'], x)

    # Head
    x = _avgpool3d(x, (2, 7, 7), (1, 1, 1))
    x = _unit3d(params['logits'], x, use_bn=False, use_relu=False)

    # Squeeze spatial dims and average over time
    x = x[:, :, 0, 0, :]
    x = jnp.mean(x, axis=1)
    return x


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _nest_params(flat):
    """Convert flat '/' separated keys into a nested dict.

    E.g. flat['Mixed_3b/b0/conv_w'] -> nested['Mixed_3b']['b0']['conv_w']
    """
    nested = {}
    for key, val in flat.items():
        parts = key.split('/')
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return nested


def _convert_pt_to_npz(pt_path, out_path):
    """Convert PyTorch I3D weights to JAX-compatible .npz format."""
    import torch
    sd = torch.load(pt_path, map_location='cpu')
    params = {}
    for key, val in sd.items():
        arr = val.numpy()
        jax_key = key.replace('.', '/')
        jax_key = jax_key.replace('/conv3d/weight', '/conv_w')
        jax_key = jax_key.replace('/conv3d/bias', '/conv_b')
        jax_key = jax_key.replace('/bn/weight', '/bn_scale')
        jax_key = jax_key.replace('/bn/bias', '/bn_offset')
        jax_key = jax_key.replace('/bn/running_mean', '/bn_mean')
        jax_key = jax_key.replace('/bn/running_var', '/bn_var')
        if 'num_batches_tracked' in jax_key:
            continue
        # Transpose conv weights: PyTorch [O, I, D, H, W] -> JAX [D, H, W, I, O]
        if jax_key.endswith('/conv_w'):
            arr = np.transpose(arr, (2, 3, 4, 1, 0))
        params[jax_key] = arr
    np.savez(out_path, **params)
    print(f"Saved {len(params)} arrays to {out_path}")


def load_i3d_pretrained(weights_path=None):
    """Load I3D params, converting from PyTorch weights on first call.

    Search order:
    1. Explicit weights_path if provided
    2. dreamer/fvd/i3d_pretrained_400.npz (next to this file)
    3. i3d_pretrained_400.pt in same directory or fvd/videogpt/ (auto-convert)

    Returns:
        Nested param dict ready for _i3d_forward().
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if weights_path is not None:
        npz_path = weights_path
    else:
        npz_path = os.path.join(this_dir, 'i3d_pretrained_400.npz')
    if not os.path.exists(npz_path):
        pt_candidates = [
            os.path.join(this_dir, 'i3d_pretrained_400.pt'),
            os.path.join(this_dir, '..', '..', 'fvd', 'videogpt', 'i3d_pretrained_400.pt'),
        ]
        pt_path = next((p for p in pt_candidates if os.path.exists(p)), None)
        if pt_path is None:
            raise FileNotFoundError(
                f"I3D weights not found at {npz_path}. "
                "Place i3d_pretrained_400.npz in dreamer/fvd/, or "
                "i3d_pretrained_400.pt in dreamer/fvd/ or fvd/videogpt/ for auto-conversion.")
        print("Converting PyTorch I3D weights to JAX format...")
        _convert_pt_to_npz(pt_path, npz_path)

    data = dict(np.load(npz_path))
    data = {k: jnp.array(v) for k, v in data.items()}
    return _nest_params(data)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(videos):
    """Preprocess videos for I3D.

    Args:
        videos: jnp.ndarray of shape [B, T, H, W, C] in uint8 [0, 255].

    Returns:
        jnp.ndarray of shape [B, T, 224, 224, C] in float32 [-1, 1].
    """
    videos = videos.astype(jnp.float32) / 255.0
    # Resize to 224x224
    b, t, h, w, c = videos.shape
    videos = jax.image.resize(videos, (b, t, 224, 224, c), method='bilinear')
    # [0, 1] -> [-1, 1]
    videos = videos * 2.0 - 1.0
    return videos


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@jax.jit
def _preprocess_and_forward(params, videos):
    """JIT-compiled: preprocess + I3D forward pass."""
    videos = preprocess(videos)
    return _i3d_forward(params, videos)


def get_fvd_logits(videos, params, bs=10):
    """Extract I3D logits from a batch of videos.

    Args:
        videos: jnp.ndarray of shape [B, T, H, W, C] in uint8 [0, 255].
        params: nested param dict from load_i3d_pretrained().
        bs: batch size for chunked inference.

    Returns:
        jnp.ndarray of shape [B, 400].
    """
    logits = []
    for i in range(0, videos.shape[0], bs):
        batch = videos[i:i + bs]
        logits.append(_preprocess_and_forward(params, batch))
    return jnp.concatenate(logits, axis=0)


# ---------------------------------------------------------------------------
# Fréchet distance
# ---------------------------------------------------------------------------

def _sqrtm_psd(mat):
    """Symmetric positive semi-definite matrix square root via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    eigvals = jnp.maximum(eigvals, 0.0)
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T


def _trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _sqrtm_psd(sigma)
    sqrt_a_sigmav_a = sqrt_sigma @ sigma_v @ sqrt_sigma
    sqrt_a_sigmav_a = (sqrt_a_sigmav_a + sqrt_a_sigmav_a.T) / 2.0
    return jnp.trace(_sqrtm_psd(sqrt_a_sigmav_a))


def _cov(m):
    """Estimate covariance matrix. m: [N, D] where rows are observations."""
    m = m - jnp.mean(m, axis=0, keepdims=True)
    return (m.T @ m) / (m.shape[0] - 1)


@jax.jit
def _frechet_distance_jit(x1, x2):
    """JIT-compiled Fréchet distance computation."""
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    m, m_w = jnp.mean(x1, axis=0), jnp.mean(x2, axis=0)
    sigma, sigma_w = _cov(x1), _cov(x2)
    mean_diff = jnp.sum((m - m_w) ** 2)
    sqrt_trace = _trace_sqrt_product(sigma, sigma_w)
    trace = jnp.trace(sigma + sigma_w) - 2.0 * sqrt_trace
    return trace + mean_diff


def frechet_distance(x1, x2):
    """Compute Fréchet distance between two sets of I3D features.

    Args:
        x1, x2: jnp.ndarray of shape [N, 400].

    Returns:
        Scalar FVD value (float).
    """
    if x1.shape[0] > 1:
        return float(_frechet_distance_jit(x1, x2))
    # Single sample: only mean difference (no covariance)
    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)
    return float(jnp.sum((jnp.mean(x1, axis=0) - jnp.mean(x2, axis=0)) ** 2))
