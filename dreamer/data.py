import io

import jax
import numpy as np
import grain
from typing import Any
import pickle
import glob
import os

import decord
decord.bridge.set_bridge("native")

from .configs import DatasetConfig


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
    """Parse MP4 video bytes using decord, random slice for Minecraft VPT dataset."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict:
        data = pickle.loads(element)

        # Decode MP4 bytes using decord (create cpu context here to avoid pickling issues with grain workers)
        # num_threads=1 to avoid oversubscription since grain already uses multiple workers
        mp4_bytes = io.BytesIO(data["video"])
        vr = decord.VideoReader(mp4_bytes, ctx=decord.cpu(0), num_threads=1)

        episode_len = len(vr)
        max_start = episode_len - self.seq_len
        start = int(rng.integers(0, max_start + 1))

        # Get frame indices for the slice
        frame_indices = list(range(start, start + self.seq_len))
        video_slice = vr.get_batch(frame_indices).asnumpy()  # (T, H, W, C)

        # Keep as float32 in [0, 255] range (consistent with CoinRun loader)
        video = video_slice.astype(np.float32)

        return {
            "videos": video,
            "actions": None,
            "rewards": None,
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
    """
    # Build array record paths based on dataset type
    if cfg.name == "minecraft_vpt":
        # Minecraft VPT: generate shard paths from index_max
        assert cfg.index_max >= 0, "index_max must be > 0 for minecraft_vpt dataset"
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

    # Build operations based on dataset type
    if cfg.name == "minecraft_vpt":
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
