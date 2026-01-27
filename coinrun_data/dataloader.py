from __future__ import annotations
import jax
import numpy as np
import grain
from typing import Any
import pickle
import os


class EpisodeLengthFilter(grain.transforms.Filter):
    """
    A Grain Filter that keeps only episodes with sufficient length.
    """

    def __init__(
        self,
        seq_len: int,
        image_h: int,
        image_w: int,
        image_c: int,
        *,
        print_filter_warnings: bool = True,
    ):
        """Initializes the filter with sequence length requirements."""
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.print_filter_warnings = print_filter_warnings

    def filter(self, element: Any) -> bool:
        """
        Filters episodes based on length.

        Args:
            element: A dictionary representing one record from the DataSource.
                     Expected to contain 'raw_video' (bytes) and 'sequence_length' (int)

        Returns:
            True if the episode has sufficient length, False otherwise.
        """
        assert isinstance(element, bytes)
        element = pickle.loads(element)

        current_episode_len = element["sequence_length"]
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
        """Initializes the transformation with processing parameters."""
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c
        self.p_include_reward = float(p_include_reward)

    def random_map(self, element: dict, rng: np.random.Generator) -> Any:
        """
        Processes a single raw episode from the data source.

        Args:
            element: A dictionary representing one record from the DataSource.
                     Expected to contain 'raw_video' (bytes) and 'sequence_length' (int)
            rng: A per-record random number generator provided by the Grain sampler.

        Returns:
            A processed video sequence as a NumPy array with shape
            (seq_len, height, width, channels) and dtype float32.
        """
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
                f"requested sequence length {self.seq_len}. This should "
                f"have been filtered out."
            )

        max_start_idx = current_episode_len - self.seq_len

        # Optionally bias slicing to include sparse rewards.
        # Assumption for CoinRun sparse reward setting: rewards are either 0.0 or 10.0.
        # If the episode contains a 10.0 reward, then with probability p_include_reward,
        # we pick a window guaranteed to include at least one rewarding timestep.
        # Load rewards once as an ndarray for all logic below
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
        actions = actions_tensor[start_idx : start_idx + self.seq_len]
        data_dict["actions"] = actions

        rewards = rewards_tensor[start_idx : start_idx + self.seq_len]
        data_dict["rewards"] = rewards

        return data_dict


def get_dataloader(
    array_record_paths: list[str],
    seq_len: int,
    global_batch_size: int,
    image_h: int,
    image_w: int,
    image_c: int,
    num_workers: int = 1,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
    p_include_reward: float = 0.0,
    *,
    print_filter_warnings: bool = True,
):
    """
    Creates a data loading pipeline using Grain.
    """
    if not array_record_paths:
        raise ValueError("array_record_paths list cannot be empty.")
    
    if isinstance(array_record_paths, str):
        # look at all the files inside of the directory that end with .array_record
        array_record_paths = [
            os.path.join(array_record_paths, f)
            for f in os.listdir(array_record_paths)
            if f.endswith(".array_record")
        ]

    num_processes = jax.process_count()

    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes} for proper sharding."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = grain.sources.ArrayRecordDataSource(array_record_paths)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        EpisodeLengthFilter(
            seq_len=seq_len,
            image_h=image_h,
            image_w=image_w,
            image_c=image_c,
            print_filter_warnings=print_filter_warnings,
        ),
        ProcessEpisodeAndSlice(
            seq_len=seq_len,
            image_h=image_h,
            image_w=image_w,
            image_c=image_c,
            p_include_reward=p_include_reward,
        ),
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