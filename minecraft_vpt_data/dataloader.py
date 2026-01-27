from __future__ import annotations
import jax
import numpy as np
import grain
import pickle


class EpisodeLengthFilter(grain.transforms.Filter):
    """Filter episodes shorter than seq_len."""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def filter(self, element: bytes) -> bool:
        data = pickle.loads(element)
        episode_len = data["video_shape"][0]
        return episode_len >= self.seq_len


class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """Parse video bytes, random slice, return dict."""

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


def get_dataloader(
    data_dir: str,
    index_max: int,
    seq_len: int,
    global_batch_size: int,
    image_h: int = 360,
    image_w: int = 640,
    image_c: int = 3,
    num_workers: int = 1,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
):
    """Create dataloader for Minecraft VPT dataset."""
    # Generate shard paths
    shard_paths = [f"{data_dir}/shard-{i:05d}.array_record" for i in range(index_max)]

    num_processes = jax.process_count()
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = grain.sources.ArrayRecordDataSource(shard_paths)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        EpisodeLengthFilter(seq_len=seq_len),
        ProcessEpisodeAndSlice(seq_len=seq_len),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]

    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=1,
    )

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=1,
        read_options=read_options,
    )

    return dataloader
