import io
import jax
import numpy as np
import grain
from typing import Any
import pickle
import glob
import os

from .configs import DatasetConfig


# ==============================================================================
# VPT Action Conversion
# ==============================================================================

VPT_BUTTON_KEYS = [
    "attack", "back", "forward", "jump", "left", "right",
    "sneak", "sprint", "use", "drop", "inventory",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9"
]


def vpt_action_to_array(action_dict: dict) -> np.ndarray:
    """Convert VPT action dict to (22,) float32 array: [camera(2), buttons(20)]."""
    arr = np.zeros(22, dtype=np.float32)
    arr[0:2] = action_dict.get("camera", [0.0, 0.0])
    buttons = action_dict.get("buttons", {})
    for i, key in enumerate(VPT_BUTTON_KEYS):
        arr[2 + i] = float(buttons.get(key, 0))
    arr[21] = float(action_dict.get("ESC", 0))
    return arr


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
        element = pickle.loads(element)

        # Handle both CoinRun and Minecraft VPT formats
        if "sequence_length" in element:
            current_episode_len = element["sequence_length"]
        elif "video_shape" in element:
            current_episode_len = element["video_shape"][0]
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


class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """
    A Grain Transformation that combines parsing, slicing, and normalizing.
    """

    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int,
        *,
        p_include_reward: float = 0.0,
    ):
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.p_include_reward = float(p_include_reward)

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

        data_dict = {"videos": seq}
        actions_tensor = np.array(element["actions"])
        data_dict["actions"] = actions_tensor[start_idx : start_idx + self.seq_len]
        data_dict["rewards"] = rewards_tensor[start_idx : start_idx + self.seq_len]

        return data_dict


# ==============================================================================
# Minecraft VPT Dataset
# ==============================================================================

class MinecraftVPTProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """Parse video bytes, random slice for Minecraft VPT dataset."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict:
        data = pickle.loads(element)

        video_shape = data["video_shape"]
        video = np.frombuffer(data["video"], dtype=np.uint8).reshape(video_shape)

        episode_len = video_shape[0]
        max_start = episode_len - self.seq_len
        start = int(rng.integers(0, max_start + 1))

        return {
            "videos": video[start : start + self.seq_len],
            "actions": None,
            "rewards": None,
        }


# ==============================================================================
# Minecraft VPT Dataset (Decord-based for MP4 bytes)
# ==============================================================================

class MinecraftVPTDecordFilter(grain.transforms.Filter):
    """Filter for Minecraft VPT records with MP4 bytes using decord."""

    def __init__(self, seq_len: int, min_frames: int = 250):
        self.seq_len = seq_len
        self.min_frames = min_frames

    def filter(self, element: bytes) -> bool:
        from decord import VideoReader

        data = pickle.loads(element)
        try:
            vr = VideoReader(io.BytesIO(data["video"]))
            return len(vr) >= max(self.seq_len, self.min_frames)
        except Exception:
            return False


class MinecraftVPTDecordProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """Process Minecraft VPT records with MP4 bytes using decord."""

    def __init__(self, seq_len: int, include_actions: bool = True):
        self.seq_len = seq_len
        self.include_actions = include_actions

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict:
        from decord import VideoReader, cpu

        data = pickle.loads(element)

        # Decode video
        vr = VideoReader(io.BytesIO(data["video"]), ctx=cpu(0))
        frame_count = len(vr)

        # Random slice
        max_start = max(0, frame_count - self.seq_len)
        start = int(rng.integers(0, max_start + 1))
        frames = vr.get_batch(range(start, start + self.seq_len)).asnumpy()

        result = {
            "videos": frames,
            "rewards": np.zeros(self.seq_len, dtype=np.float32),
        }

        if self.include_actions:
            action_list = pickle.loads(data["actions"])
            action_slice = action_list[start : start + self.seq_len]
            result["actions"] = np.stack([vpt_action_to_array(a) for a in action_slice])
        else:
            result["actions"] = None

        return result


# ==============================================================================
# Factory
# ==============================================================================

def make_iterator(
    cfg: DatasetConfig,
    num_workers: int = 22,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
    print_filter_warnings: bool = False,
    use_decord: bool = False,
):
    """
    Creates a data loading pipeline using Grain from a DatasetConfig.

    Args:
        use_decord: If True, use decord to decode MP4 bytes on-the-fly for minecraft_vpt.
            This is required for data preprocessed with preprocess_minecraft.py.
    """
    # Build array record paths based on dataset type
    if cfg.name == "minecraft_vpt":
        # Minecraft VPT: generate shard paths from index_max
        assert cfg.index_max >= 0, "index_max must be > 0 for minecraft_vpt dataset"
        array_record_paths = [f"{cfg.array_record_path}/shard-{i:06d}.array_record" for i in range(cfg.index_max)]
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

    # Build operations based on dataset type
    if cfg.name == "minecraft_vpt":
        if use_decord:
            # Decord-based pipeline for MP4 bytes (from preprocess_minecraft.py)
            operations = [
                MinecraftVPTDecordFilter(seq_len=cfg.T),
                MinecraftVPTDecordProcessEpisodeAndSlice(
                    seq_len=cfg.T,
                    include_actions=True,
                ),
                grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
            ]
        else:
            # Legacy pipeline for raw pixel bytes with video_shape metadata
            operations = [
                EpisodeLengthFilter(
                    seq_len=cfg.T,
                    print_filter_warnings=print_filter_warnings,
                ),
                MinecraftVPTProcessEpisodeAndSlice(seq_len=cfg.T),
                grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
            ]
    else:
        operations = [
            EpisodeLengthFilter(
                seq_len=cfg.T,
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessEpisodeAndSlice(
                seq_len=cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                p_include_reward=cfg.p_include_reward,
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
