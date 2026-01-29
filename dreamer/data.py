import copy
import io

import jax
import msgpack
import numpy as np
import grain
from typing import Any, Tuple
import pickle
import os

import decord
decord.bridge.set_bridge("native")

from .configs import DatasetConfig
from .actions import Actions


# ==============================================================================
# CoinRun Dataset
# ==============================================================================

class EpisodeLengthFilter(grain.transforms.Filter):
    """
    A Grain Filter that keeps only episodes with sufficient length.
    """

    def __init__(
        self,
        seq_len: int,
        *,
        print_filter_warnings: bool = True,
    ):
        self.seq_len = seq_len
        self.print_filter_warnings = print_filter_warnings

    def filter(self, element: Any) -> bool:
        assert isinstance(element, bytes)
        data = pickle.loads(element)

        # Handle both CoinRun and Minecraft VPT formats
        if "sequence_length" in data:
            current_episode_len = data["sequence_length"]
        elif "video_shape" in data:
            current_episode_len = data["video_shape"][0]
        else:
            raise ValueError("Unknown episode format: missing 'sequence_length' or 'video_shape'")

        if current_episode_len < self.seq_len:
            if self.print_filter_warnings:
                print(
                    f"Filtering out episode with length {current_episode_len}, which is "
                    f"shorter than the requested sequence length {self.seq_len}."
                )
            return False

        return True


class CoinrunProcessEpisodeAndSlice(grain.transforms.RandomMap):
    # TODO: consolidate with other process and slice classes
    """
    A Grain Transformation that combines parsing, slicing, and normalizing.
    """

    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int,
        padding_h: tuple[int, int],
        padding_w: tuple[int, int],
        *,
        p_include_reward: float = 0.0,
        patch_size: int,
    ):
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.p_include_reward = float(p_include_reward)
        self.padding_h = padding_h
        self.padding_w = padding_w

        assert sum(padding_h, image_h) % patch_size == 0
        assert sum(padding_w, image_w) % patch_size == 0

    def random_map(self, element: dict, rng: np.random.Generator) -> Any:
        assert isinstance(element, bytes)
        element = pickle.loads(element)

        video_shape = (
            element["sequence_length"],
            self.image_h,
            self.image_w,
            self.image_c,
        )
        episode_tensor = np.frombuffer(element["raw_video"], dtype=np.uint8)
        episode_tensor = episode_tensor.reshape(video_shape)

        current_episode_len = episode_tensor.shape[0]
        if current_episode_len < self.seq_len:
            raise ValueError(
                f"Episode length {current_episode_len} is shorter than "
                f"requested sequence length {self.seq_len}."
            )

        max_start_idx = current_episode_len - self.seq_len
        rewards_tensor = np.array(element["rewards"])

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

        seq = episode_tensor[start_idx : start_idx + self.seq_len]

        seq = np.pad(
            seq,
            ((0, 0), self.padding_h, self.padding_w, (0, 0)),
            mode='constant',
            constant_values=0
        )

        actions_tensor = np.array(element["actions"])
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
# Minecraft VPT Dataset
# ==============================================================================

class MinecraftVPTProcessEpisodeAndSlice(grain.transforms.RandomMap):
    # TODO: consolidate with other process and slice classes
    """Parse MP4 video bytes using decord, random slice for Minecraft VPT dataset."""

    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int,
        padding_h: tuple[int, int],
        padding_w: tuple[int, int],
        *,
        patch_size: int,
    ):
        self.seq_len = seq_len
        self.padding_h = padding_h
        self.padding_w = padding_w

        assert sum(padding_h, image_h) % patch_size == 0
        assert sum(padding_w, image_w) % patch_size == 0

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict[str, Any]:
        data = pickle.loads(element)

        # Decode MP4 bytes using decord (create cpu context here to avoid pickling issues with grain workers)
        # num_threads=1 to avoid oversubscription since grain already uses multiple workers
        mp4_bytes = io.BytesIO(data["video"])

        cpu_idx = int(rng.integers(0, 2))
        vr = decord.VideoReader(mp4_bytes, ctx=decord.cpu(cpu_idx), num_threads=1)

        episode_len = len(vr)
        max_start = episode_len - self.seq_len
        start = int(rng.integers(0, max_start + 1))

        # Get frame indices for the slice
        frame_indices = list(range(start, start + self.seq_len))
        video_slice = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

        # Keep as float32 in [0, 255] range (consistent with CoinRun loader)
        video = video_slice.astype(np.float32)

        video = np.pad(
            video,
            ((0, 0), self.padding_h, self.padding_w, (0, 0)),
            mode='constant',
            constant_values=0
        )

        return {
            "videos": video,
            "actions": Actions(binary=None, categorical=None, continuous=None),  # FIXME: no actions returned!!
            "rewards": None,
        }


# ==============================================================================
# Pre-tokenized Latent Dataset
# ==============================================================================

def _decode_value(value):
    """Decode a single value, handling arrays and nested dicts."""
    if isinstance(value, dict):
        if value.get("_type") == "ndarray":
            shape = tuple(value["shape"])
            dtype = value["dtype"]
            return np.frombuffer(value["data"], dtype=dtype).reshape(shape)
        else:
            return {k: _decode_value(v) for k, v in value.items()}
    return value


def deserialize_latent_record(data: bytes) -> dict:
    """Deserialize latent record from msgpack format."""
    encoded = msgpack.unpackb(data, raw=False)
    return {k: _decode_value(v) for k, v in encoded.items()}


class LatentEpisodeLengthFilter(grain.transforms.Filter):
    """Filter latent episodes by sequence length."""
    def __init__(self, seq_len: int, *, print_filter_warnings: bool = True):
        self.seq_len = seq_len
        self.print_filter_warnings = print_filter_warnings

    def filter(self, element: bytes) -> bool:
        data = deserialize_latent_record(element)
        episode_len = data["latents"].shape[0]
        if episode_len < self.seq_len:
            if self.print_filter_warnings:
                print(f"Filtering latent episode: {episode_len} < {self.seq_len}")
            return False
        return True


class ProcessLatentAndSlice(grain.transforms.RandomMap):
    # TODO: consolidate with other process and slice classes
    """Random slice pre-tokenized latent episodes."""
    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict:
        data = deserialize_latent_record(element)
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


# ==============================================================================
# Factory
# ==============================================================================

def make_iterator(
    cfg: DatasetConfig,
    num_workers: int = 64,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
    print_filter_warnings: bool = False,
):
    """
    Creates a data loading pipeline using Grain from a DatasetConfig.
    
    Args:
        cfg: Dataset configuration
        num_workers: Number of worker processes
        prefetch_buffer_size: Prefetch buffer size
        seed: Random seed
        print_filter_warnings: Whether to print filter warnings
    """
    # Build array record paths based on dataset type and data type
    use_latent_data = cfg.data_type == "latent"

    if use_latent_data or cfg.name == "minecraft_vpt":
        # Latent or Minecraft VPT: generate shard paths from index_max
        assert cfg.index_max >= 0, "index_max must be > 0 for minecraft_vpt or latent dataset"
        array_record_paths = [f"{cfg.array_record_path}/shard-{i:05d}.array_record" for i in range(cfg.index_max)]
    else:
        # CoinRun: discover files in directory
        array_record_paths = cfg.array_record_path
        if not array_record_paths:
            raise ValueError("array_record_path cannot be empty.")
        if isinstance(array_record_paths, str):
            if os.path.isdir(array_record_paths):
                array_record_paths = [
                    os.path.join(array_record_paths, f)
                    for f in os.listdir(array_record_paths)
                    if f.endswith(".array_record")
                ]
            else:
                array_record_paths = [array_record_paths]

    num_processes = jax.process_count()

    if cfg.B % num_processes != 0:
        raise ValueError(
            f"Global batch size {cfg.B} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = cfg.B // num_processes

    source = grain.sources.ArrayRecordDataSource(array_record_paths)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    # Build operations based on dataset type and data type
    if use_latent_data:
        # Pre-tokenized latent data path
        operations = [
            LatentEpisodeLengthFilter(
                seq_len=cfg.T,
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessLatentAndSlice(seq_len=cfg.T),
            grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
        ]
    elif cfg.name == "minecraft_vpt":
        operations = [
            EpisodeLengthFilter(
                seq_len=cfg.T,
                print_filter_warnings=print_filter_warnings,
            ),
            MinecraftVPTProcessEpisodeAndSlice(
                seq_len=cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                patch_size=cfg.patch_size,
            ),
            grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
        ]
    else:
        operations = [
            EpisodeLengthFilter(
                seq_len=cfg.T,
                print_filter_warnings=print_filter_warnings,
            ),
            CoinrunProcessEpisodeAndSlice(
                seq_len=cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                p_include_reward=cfg.p_include_reward,
                patch_size=cfg.patch_size,
            ),
            grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
        ]

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(
            prefetch_buffer_size=prefetch_buffer_size,
            num_threads=1,
        ),
    )

    return dataloader


def make_dual_iterators(
    cfg: DatasetConfig,
    short_T: int,
    long_T: int,
    num_workers: int = 22,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
    print_filter_warnings: bool = False,
) -> Tuple[grain.DataLoader, grain.DataLoader]:
    """Create separate iterators for short and long sequences (alternating batch lengths)."""
    # Create config for short sequences
    cfg_short = copy.copy(cfg)
    cfg_short.T = short_T

    # Create config for long sequences
    cfg_long = copy.copy(cfg)
    cfg_long.T = long_T

    # Use different seeds to avoid correlation between iterators
    short_loader = make_iterator(
        cfg_short,
        num_workers=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
        seed=seed,
        print_filter_warnings=print_filter_warnings,
    )
    long_loader = make_iterator(
        cfg_long,
        num_workers=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
        seed=seed + 1,  # Different seed for variety
        print_filter_warnings=print_filter_warnings,
    )

    return short_loader, long_loader
