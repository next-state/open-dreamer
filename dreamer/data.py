import jax
import numpy as np
import grain
from typing import Any
import pickle
import os

import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types

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

class _DALICPUDecoder:
    """Single-video DALI CPU decoder with controlled threading."""

    def __init__(self, num_threads: int = 1):
        @pipeline_def(
            batch_size=1,
            num_threads=num_threads,
            device_id=None,  # CPU-only
            exec_pipelined=False,
            exec_async=False,
        )
        def decode_pipe():
            encoded = fn.external_source(name="video", dtype=types.UINT8)
            frames_idx = fn.external_source(name="frames", dtype=types.INT32)
            video = fn.experimental.decoders.video(encoded, device="cpu", frames=frames_idx)
            return video

        self.pipe = decode_pipe()
        self.pipe.build()

    def decode(self, mp4_bytes: bytes, start_frame: int, seq_len: int) -> np.ndarray:
        video_array = np.frombuffer(mp4_bytes, dtype=np.uint8)
        frame_indices = np.arange(start_frame, start_frame + seq_len, dtype=np.int32)

        self.pipe.feed_input("video", [video_array])
        self.pipe.feed_input("frames", [frame_indices])
        output = self.pipe.run()
        return output[0].as_array()[0]  # Remove batch dim


class MinecraftVPTProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """Parse MP4 video bytes using DALI CPU decode, random slice for Minecraft VPT dataset.

    Uses DALI's CPU video decoder with controlled threading to avoid thread explosion
    when running with many grain workers.
    """

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self._decoder = None  # Lazy init in worker process

    def _get_decoder(self) -> _DALICPUDecoder:
        if self._decoder is None:
            self._decoder = _DALICPUDecoder(num_threads=1)
        return self._decoder

    def random_map(self, element: bytes, rng: np.random.Generator) -> dict:
        data = pickle.loads(element)
        episode_len = data["video_shape"][0]

        max_start = episode_len - self.seq_len
        start = int(rng.integers(0, max_start + 1))

        decoder = self._get_decoder()
        video = decoder.decode(data["video"], start, self.seq_len)

        # Keep as float32 in [0, 255] range (consistent with CoinRun loader)
        return {
            "videos": video.astype(np.float32),
            "actions": None,
            "rewards": None,
        }


# ==============================================================================
# Factory
# ==============================================================================

def make_iterator(
    cfg: DatasetConfig,
    num_workers: int = 64,
    prefetch_buffer_size: int = 4,
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


# ==============================================================================
# DALI GPU-Accelerated Video Loading (Alternative)
# ==============================================================================


class ArrayRecordVideoSource:
    """Random-access source for DALI parallel external_source.

    Uses lazy initialization to avoid pickling issues with ArrayRecordDataSource.
    """

    def __init__(self, array_record_paths: list, seq_len: int, valid_indices: list, seed: int = 42):
        self.array_record_paths = array_record_paths
        self.seq_len = seq_len
        self.valid_indices = valid_indices
        self.seed = seed
        self._source = None  # Lazy init in worker

    def _get_source(self):
        """Lazy init the ArrayRecordDataSource in the worker process."""
        if self._source is None:
            self._source = grain.sources.ArrayRecordDataSource(self.array_record_paths)
        return self._source

    def __call__(self, sample_info):
        """Called by DALI for each sample. Must be thread-safe for parallel mode."""
        source = self._get_source()

        # Use sample index to pick a valid episode (with wrap-around)
        idx = self.valid_indices[sample_info.idx_in_epoch % len(self.valid_indices)]

        # Load data
        data = pickle.loads(source[idx])
        episode_len = data["video_shape"][0]

        # Deterministic random start based on iteration for reproducibility
        rng = np.random.default_rng(self.seed + sample_info.iteration * 1000 + sample_info.idx_in_batch)
        max_start = episode_len - self.seq_len
        start = int(rng.integers(0, max_start + 1))

        video_bytes = np.frombuffer(data["video"], dtype=np.uint8)
        # Return video bytes and frame indices to extract
        frame_indices = np.arange(start, start + self.seq_len, dtype=np.int32)
        return video_bytes, frame_indices


def _find_valid_indices(array_record_paths: list, seq_len: int) -> list:
    """Pre-scan to find episodes that are long enough."""
    source = grain.sources.ArrayRecordDataSource(array_record_paths)
    num_samples = len(source)
    print(f"Scanning {num_samples} episodes for valid lengths...", flush=True)
    valid = []
    for i in range(num_samples):
        data = pickle.loads(source[i])
        if data["video_shape"][0] >= seq_len:
            valid.append(i)
    print(f"Found {len(valid)} valid episodes", flush=True)
    return valid


@pipeline_def
def dali_video_pipeline(external_source, seq_len: int, parallel: bool = True):
    """DALI pipeline for GPU-accelerated video decoding."""
    # Get encoded videos and frame indices from external source
    encoded, frame_indices = fn.external_source(
        source=external_source,
        num_outputs=2,
        dtype=[types.UINT8, types.INT32],
        batch=False,
        parallel=parallel,
        prefetch_queue_depth=4 if parallel else 1,
    )

    # GPU-accelerated video decoding with frame selection
    # Use 'frames' parameter to specify exact frame indices to extract
    videos = fn.experimental.decoders.video(
        encoded,
        device='mixed',  # NVDEC hardware decode
        frames=frame_indices,
    )

    # Cast to float32 [0, 255] range
    videos = fn.cast(videos, dtype=types.FLOAT)

    return videos


class DALIIteratorWrapper:
    """Wrapper around DALI pipeline that yields JAX-compatible dicts."""

    def __init__(self, pipe):
        self.pipe = pipe

    def __iter__(self):
        return self

    def __next__(self):
        output = self.pipe.run()
        # output[0] is TensorListGPU, convert to numpy via as_cpu()
        videos_tl = output[0].as_cpu()
        videos_np = np.stack([videos_tl.at(i) for i in range(len(videos_tl))])
        return {"videos": jax.device_put(videos_np.astype(np.float32))}


def make_dali_iterator(
    cfg: DatasetConfig,
    num_threads: int = 4,
    device_id: int = 0,
    seed: int = 42,
    parallel: bool = True,
):
    """Create DALI-based video loader with GPU decoding."""
    array_record_paths = [
        f"{cfg.array_record_path}/shard-{i:05d}.array_record"
        for i in range(cfg.index_max)
    ]

    # Pre-compute valid indices (done once in main process)
    valid_indices = _find_valid_indices(array_record_paths, cfg.T)

    external_source = ArrayRecordVideoSource(
        array_record_paths=array_record_paths,
        seq_len=cfg.T,
        valid_indices=valid_indices,
        seed=seed,
    )

    pipe = dali_video_pipeline(
        external_source=external_source,
        seq_len=cfg.T,
        parallel=parallel,
        batch_size=cfg.B,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=2,  # Overlap I/O with compute
    )
    pipe.build()

    return DALIIteratorWrapper(pipe)
