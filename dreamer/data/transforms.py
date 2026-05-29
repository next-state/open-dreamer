"""Unified Grain transforms for all dataset types.

Provides flexible, reusable transforms that handle:
- Episode length filtering with auto-detection of format
- CoinRun episode processing with reward biasing
- Minecraft VPT episode processing with MP4 decoding
- Pre-tokenized latent episode processing
- Action dataclass creation
"""

import io
import os
import pickle
from typing import Any
import jax
import jax.numpy as jnp
import grain
import numpy as np

from ..actions import Actions, parse_action_dicts
from .serialization import deserialize_msgpack_record

try:
    import decord
except ImportError:
    decord = None


def _require_decord():
    if decord is None:
        raise ImportError(
            "decord is required for Minecraft VPT video decoding but is not installed "
            "for this platform."
        )
    decord.bridge.set_bridge("native")
    return decord


# ==============================================================================
# Length Filtering
# ==============================================================================

class EpisodeLengthFilter(grain.transforms.Filter):
    """Universal length filter with auto-detection of episode format.

    Supports:
    - CoinRun: {"sequence_length": int}
    - Minecraft VPT: {"video_shape": (T, H, W, C)}
    - Latent: {"latents": (T, n_latents, d_bottleneck)}
    """

    def __init__(
        self,
        seq_len: int,
        *,
        format_hint: str = "auto",
        print_filter_warnings: bool = True,
    ):
        """Initialize episode length filter.

        Args:
            seq_len: Minimum sequence length required
            format_hint: Format hint ("auto", "coinrun", "vpt", "latent")
            print_filter_warnings: Whether to print warnings for filtered episodes
        """
        self.seq_len = seq_len
        self.format_hint = format_hint
        self.print_filter_warnings = print_filter_warnings

    def filter(self, element: bytes) -> bool:
        """Filter episodes by length.

        Args:
            element: Serialized episode bytes

        Returns:
            True if episode is long enough, False otherwise
        """
        # Try to determine format and extract length
        if self.format_hint == "latent":
            data = deserialize_msgpack_record(element)
            episode_len = data["latents"].shape[0]
        else:
            # CoinRun and VPT use pickle
            data = pickle.loads(element)

            if self.format_hint == "coinrun":
                episode_len = data["sequence_length"]
            elif self.format_hint == "vpt":
                episode_len = data["video_shape"][0]
            else:  # auto-detect
                if "sequence_length" in data:
                    episode_len = data["sequence_length"]
                elif "video_shape" in data:
                    episode_len = data["video_shape"][0]
                else:
                    raise ValueError(
                        "Unknown episode format: missing 'sequence_length' or 'video_shape'"
                    )

        if episode_len < self.seq_len:
            if self.print_filter_warnings:
                print(
                    f"Filtering out episode with length {episode_len}, which is "
                    f"shorter than the requested sequence length {self.seq_len}."
                )
            return False

        return True


# ==============================================================================
# CoinRun Processing
# ==============================================================================

class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """Process CoinRun episodes with optional padding and reward biasing.

    Parses raw video bytes, randomly slices to seq_len, applies padding,
    and optionally biases slicing toward timesteps with rewards.
    """

    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int,
        *,
        padding_h: tuple[int, int] = (0, 0),
        padding_w: tuple[int, int] = (0, 0),
        p_include_reward: float = 0.0,
        patch_size: int | None = None,
    ):
        """Initialize CoinRun processor.

        Args:
            seq_len: Target sequence length
            image_h: Image height
            image_w: Image width
            image_c: Image channels
            padding_h: Padding for height (top, bottom)
            padding_w: Padding for width (left, right)
            p_include_reward: Probability of biasing slice toward rewards
            patch_size: Patch size for padding validation (required if padding used)
        """
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.p_include_reward = float(p_include_reward)
        self.padding_h = padding_h
        self.padding_w = padding_w

        # Validate padding alignment with patch_size
        if patch_size is not None:
            assert (sum(self.padding_h) + image_h) % patch_size == 0, \
                f"Height {image_h} + padding {self.padding_h} must be divisible by patch_size {patch_size}"
            assert (sum(self.padding_w) + image_w) % patch_size == 0, \
                f"Width {image_w} + padding {self.padding_w} must be divisible by patch_size {patch_size}"

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict[str, Any]:
        """Process and randomly slice CoinRun episode.

        Args:
            element: Pickled episode bytes
            rng: Random number generator

        Returns:
            Dictionary with videos, actions, and rewards
        """
        data = pickle.loads(element)

        # Reshape raw video bytes
        video_shape = (
            data["sequence_length"],
            self.image_h,
            self.image_w,
            self.image_c,
        )
        episode_tensor = np.frombuffer(data["raw_video"], dtype=np.uint8)
        episode_tensor = episode_tensor.reshape(video_shape)

        current_episode_len = episode_tensor.shape[0]
        if current_episode_len < self.seq_len:
            raise ValueError(
                f"Episode length {current_episode_len} is shorter than "
                f"requested sequence length {self.seq_len}."
            )

        max_start_idx = current_episode_len - self.seq_len
        rewards_tensor = np.array(data["rewards"])

        # Optional reward-biased slicing
        start_idx = None
        if self.p_include_reward > 0.0 and rng.random() < self.p_include_reward:
            reward_ts = np.flatnonzero(rewards_tensor > 0)
            if reward_ts.size > 0:
                t = int(rng.choice(reward_ts))
                start_min = max(0, t - (self.seq_len - 1))
                start_max = min(t, max_start_idx)
                start_idx = int(rng.integers(start_min, start_max + 1))

        if start_idx is None:
            start_idx = int(rng.integers(0, max_start_idx + 1))

        # Slice episode
        seq = episode_tensor[start_idx : start_idx + self.seq_len]

        # Apply padding
        seq = np.pad(
            seq,
            ((0, 0), self.padding_h, self.padding_w, (0, 0)),
            mode='constant',
            constant_values=0
        )

        actions_tensor = np.array(data["actions"])
        return {
            "videos": seq,
            "actions": Actions(
                binary=None,
                categorical=actions_tensor[start_idx : start_idx + self.seq_len],
                continuous=None,
            ),
            "rewards": rewards_tensor[start_idx : start_idx + self.seq_len],
        }


# ==============================================================================
# Minecraft VPT Processing
# ==============================================================================

class ProcessMinecraftEpisodeAndSlice(grain.transforms.RandomMap):
    """Process Minecraft VPT MP4 episodes with optional padding.

    Decodes MP4 bytes using decord and either:
    - Randomly slices to seq_len for training
    - Returns full episode for tokenization (if full_episode=True)
    """

    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int,
        *,
        padding_h: tuple[int, int] = (0, 0),
        padding_w: tuple[int, int] = (0, 0),
        patch_size: int | None = None,
        full_episode: bool = False,
        decoder_threads: int = 1,
        cast_to_float32: bool = True,
        return_actions: bool = False,
    ):
        """Initialize Minecraft VPT processor.

        Args:
            seq_len: Target sequence length (ignored if full_episode=True)
            image_h: Image height
            image_w: Image width
            image_c: Image channels
            padding_h: Padding for height (top, bottom)
            padding_w: Padding for width (left, right)
            patch_size: Patch size for padding validation (required if padding used)
            full_episode: If True, return full episode without slicing (for tokenization)
            decoder_threads: Number of decoding threads per worker process
            cast_to_float32: Whether to cast decoded video to float32
        """
        self.seq_len = seq_len
        self.padding_h = tuple(padding_h) if isinstance(padding_h, list) else padding_h
        self.padding_w = tuple(padding_w) if isinstance(padding_w, list) else padding_w
        self.full_episode = full_episode
        self.decoder_threads = max(1, int(decoder_threads))
        self.cast_to_float32 = bool(cast_to_float32)
        self.return_actions = return_actions
        

        # Validate padding alignment with patch_size
        if patch_size is not None:
            assert (sum(self.padding_h) + image_h) % patch_size == 0, \
                f"Height {image_h} + padding {self.padding_h} must be divisible by patch_size {patch_size}"
            assert (sum(self.padding_w) + image_w) % patch_size == 0, \
                f"Width {image_w} + padding {self.padding_w} must be divisible by patch_size {patch_size}"

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict[str, Any]:
        """Process and optionally slice Minecraft VPT episode.

        Args:
            element: Pickled episode bytes
            rng: Random number generator

        Returns:
            Dictionary with videos and actions
        """
        data = pickle.loads(element)
        decord_module = _require_decord()

        # Decode MP4 bytes using decord
        mp4_bytes = io.BytesIO(data["video"])
        vr = decord_module.VideoReader(
            mp4_bytes, ctx=decord_module.cpu(0), num_threads=self.decoder_threads
        )

        episode_len = len(vr)

        if self.full_episode:
            video = vr.get_batch(list(range(episode_len))).asnumpy()
            actions = parse_action_dicts(data.get("actions")).to_dict()
        else:
            max_start = episode_len - self.seq_len
            start = int(rng.integers(0, max_start + 1))
            frame_indices = list(range(start, start + self.seq_len))
            video = vr.get_batch(frame_indices).asnumpy()
            if self.return_actions:
                all_actions = parse_action_dicts(data.get("actions"))
                actions = all_actions[start:start + self.seq_len]
            else:
                actions = Actions(binary=None, categorical=None, continuous=None) # TODO: might be better to pass None at this point, but i'm keeping it in not to break any backward compatibility
        # For tokenization pipelines, keeping uint8 here dramatically reduces
        # multiprocessing/shared-memory pressure; cast later on-device.
        if self.cast_to_float32:
            video = video.astype(np.float32, copy=False)

        # Apply padding
        video = np.pad(
            video,
            ((0, 0), self.padding_h, self.padding_w, (0, 0)),
            mode='constant',
            constant_values=0
        )

        result = {
            "videos": video,
            "actions": actions,
            "rewards": None,
        }
        if self.full_episode:
            result["source"] = data.get("source")
        return result


# ==============================================================================
# Latent Processing
# ==============================================================================

class ProcessLatentAndSlice(grain.transforms.RandomMap):
    """Random slice pre-tokenized latent episodes."""

    def __init__(self, seq_len: int):
        """Initialize latent processor.

        Args:
            seq_len: Target sequence length
        """
        self.seq_len = seq_len

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict[str, Any]:
        """Process and randomly slice latent episode.

        Args:
            element: Msgpack-encoded episode bytes
            rng: Random number generator

        Returns:
            Dictionary with latents and actions
        """
        data = deserialize_msgpack_record(element)
        latents = data["latents"]  # (T, n_latents, d_bottleneck)
        actions = data["actions"]  # dict with action arrays

        episode_len = latents.shape[0]
        max_start = episode_len - self.seq_len
        start = int(rng.integers(0, max_start + 1))
        end = start + self.seq_len

        return {
            "latents": latents[start:end].astype(np.float32),
            "actions": Actions.from_dict(actions)[start:end],
        }


class CastDtype(grain.transforms.Map):
    """Cast floating-point arrays to a specified dtype."""

    DTYPE_MAP = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,  # numpy doesn't support bfloat16, cast later in JAX
    }

    def __init__(self, dtype: str):
        self.dtype_str = dtype
        self.dtype = self.DTYPE_MAP.get(dtype, np.float32)

    def _cast_array(self, arr):
        if arr is None:
            return None
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(self.dtype)
        return arr

    def map(self, element):
        import jax
        return jax.tree.map(self._cast_array, element)
