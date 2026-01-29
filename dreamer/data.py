import copy
import io
import pickle
from typing import Any

import decord
import grain
import jax
import numpy as np
from grain._src.python.dataset import dataset as grain_dataset
from grain.experimental import device_put
from grain.transforms import Batch

from dreamer.actions import Actions
from dreamer.configs import DatasetConfig, DataloaderConfig
from dreamer.data.path_utils import build_dataset_paths
from dreamer.data.serialization import deserialize_msgpack_record
from dreamer.data.transforms import (
    CastDtype,
    CreateActions,
    EpisodeLengthFilter,
    ProcessEpisodeAndSlice,
    ProcessLatentAndSlice,
    ProcessMinecraftEpisodeAndSlice,
)


"""Unified Grain transforms for all dataset types.

Provides flexible, reusable transforms that handle:
- Episode length filtering with auto-detection of format
- CoinRun episode processing with reward biasing
- Minecraft VPT episode processing with MP4 decoding
- Pre-tokenized latent episode processing
- Action dataclass creation
"""

decord.bridge.set_bridge("native")


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
        """
        self.seq_len = seq_len
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.full_episode = full_episode

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

        # Decode MP4 bytes using decord
        mp4_bytes = io.BytesIO(data["video"])
        cpu_idx = int(rng.integers(0, 2))
        vr = decord.VideoReader(mp4_bytes, ctx=decord.cpu(cpu_idx), num_threads=1)

        episode_len = len(vr)

        if self.full_episode:
            # Return full episode for tokenization
            video = vr.get_batch(list(range(episode_len))).asnumpy()
        else:
            # Random slice for training
            max_start = episode_len - self.seq_len
            start = int(rng.integers(0, max_start + 1))
            frame_indices = list(range(start, start + self.seq_len))
            video = vr.get_batch(frame_indices).asnumpy()

        # Keep as float32 in [0, 255] range (consistent with CoinRun)
        video = video.astype(np.float32)

        # Apply padding
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


# ==============================================================================
# Action Processing
# ==============================================================================

class CreateActions(grain.transforms.Map):
    """Convert batched action arrays into Actions dataclass."""

    def map(self, batch: dict) -> dict:
        """Convert action arrays to Actions dataclass.

        Args:
            batch: Batch dictionary with actions_* keys

        Returns:
            Batch with actions field as Actions dataclass
        """
        batch["actions"] = Actions(
            binary=batch.pop("actions_binary", None),
            categorical=batch.pop("actions_categorical", None),
            continuous=batch.pop("actions_continuous", None),
        )
        return batch


class NumpyToJax(grain.transforms.Map):
    """Convert numpy arrays to JAX arrays."""

    def map(self, batch: dict) -> dict:
        """Convert numpy arrays in batch to JAX arrays.

        Args:
            batch: Batch dictionary with numpy arrays

        Returns:
            Batch with JAX arrays
        """
        import jax.numpy as jnp

        result = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                result[key] = jnp.array(value)
            elif hasattr(value, '__dict__'):  # Handle dataclasses like Actions
                result[key] = value
            else:
                result[key] = value
        return result


# ==============================================================================
# DataLoader to IterDataset Wrapper
# ==============================================================================

class DataLoaderIteratorWrapper(grain_dataset.IterDataset):
    """Wraps a DataLoader iterator as a DatasetIterator for use with grain.experimental.device_put."""
    
    def __init__(self, dataloader):
        super().__init__()
        self._iterator = iter(dataloader)
        self._count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self._count += 1
        return next(self._iterator)
        
    def get_state(self):
        # Basic state - just track count for debugging
        # Full checkpointing would need access to underlying dataloader state
        return {"count": self._count}
    
    def set_state(self, state):
        self._count = state.get("count", 0)

# ==============================================================================
# Factory
# ==============================================================================

def make_iterator(
    cfg: DatasetConfig,
    *,
    seed: int = 42,
    print_filter_warnings: bool = False,
    device = None,
):
    """Creates a data loading pipeline using Grain from a DatasetConfig.

    Args:
        cfg: Dataset configuration
        dataloader_cfg: Dataloader configuration (defaults to cfg.dataloader_cfg if None)
        seed: Random seed
        print_filter_warnings: Whether to print filter warnings
        device: Device for prefetching
    """
    dataloader_cfg = cfg.dataloader_cfg
    
    num_workers = dataloader_cfg.num_workers
    prefetch_buffer_size = dataloader_cfg.prefetch_buffer_size
    device_prefetch_buffer_size = dataloader_cfg.device_prefetch_buffer_size
    # Build array record paths using utility
    use_latent_data = cfg.data_type == "latent"
    dataset_type = "latent" if use_latent_data else cfg.name

    array_record_paths = build_dataset_paths(
        cfg.array_record_path,
        dataset_type=dataset_type,
        index_max=cfg.index_max if use_latent_data or cfg.name == "minecraft_vpt" else None,
    )

    num_processes = jax.process_count()

    if dataloader_cfg.B % num_processes != 0:
        raise ValueError(
            f"Global batch size {dataloader_cfg.B} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = dataloader_cfg.B // num_processes

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
        operations = [ProcessLatentAndSlice(seq_len=dataloader_cfg.T)]
    elif cfg.name == "minecraft_vpt":
        operations = [
            EpisodeLengthFilter(
                seq_len=dataloader_cfg.T,
                format_hint="vpt",
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessMinecraftEpisodeAndSlice(
                seq_len=dataloader_cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                patch_size=cfg.patch_size,
            )
        ]
    else:
        operations = [
            EpisodeLengthFilter(
                seq_len=dataloader_cfg.T,
                format_hint="coinrun",
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessEpisodeAndSlice(
                seq_len=dataloader_cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                p_include_reward=cfg.p_include_reward,
                patch_size=cfg.patch_size,
            ),
        ]
            
    common_ops = [Batch(batch_size=per_process_batch_size, drop_remainder=True), CreateActions(), CastDtype(dataloader_cfg.dtype)]
    operations = operations + common_ops

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=1,
        read_options=grain.ReadOptions(
            prefetch_buffer_size=prefetch_buffer_size if device is None else 1,
            num_threads=1,
        ),
    )
    
    if device is None:
        return iter(dataloader)
    
    # Wrap DataLoader as IterDataset for compatibility with grain.experimental.device_put
    iter_dataset = DataLoaderIteratorWrapper(dataloader)
    iter_dataset = device_put(
        iter_dataset,
        device,
        cpu_buffer_size=prefetch_buffer_size,
        device_buffer_size=device_prefetch_buffer_size if prefetch_buffer_size is not None else prefetch_buffer_size,
    )
    
    return iter_dataset




class AlternatingIterator(grain.IterDataset):
    def __init__(self, short_iterator, long_iterator, long_ratio: float, seed: int) -> None:
        self.short_iterator = short_iterator
        self.long_iterator = long_iterator
        self.long_ratio = long_ratio
        self._rng_key = jax.random.key(seed)

    def __next__(self):
        self._rng_key, subkey = jax.random.split(self._rng_key)
        if float(jax.random.uniform(subkey)) < self.long_ratio:
            return True, next(self.long_iterator)
        return False, next(self.short_iterator)

    def __iter__(self):
        return self

def make_dual_iterator(
    cfg: DatasetConfig,
    *,
    seed: int = 42,
    print_filter_warnings: bool = False,
    device = None,
    ) -> grain.IterDataset:
    """Create alternating iterator over short and long sequences.
    
    Args:
        cfg: Dataset configuration
        dataloader_cfg: Dataloader configuration (defaults to cfg.dataloader_cfg if None)
        seed: Random seed
        print_filter_warnings: Whether to print filter warnings
        device: Device for prefetching
    """
    dataloader_cfg = cfg.dataloader_cfg
    
    short_T = dataloader_cfg.short_T
    long_T = dataloader_cfg.long_T
    long_ratio = dataloader_cfg.long_ratio
    num_workers = dataloader_cfg.num_workers
    prefetch_buffer_size = dataloader_cfg.prefetch_buffer_size
    device_prefetch_buffer_size = dataloader_cfg.device_prefetch_buffer_size
    
    if short_T == long_T:
        iterator = make_iterator(
            cfg,
            seed=seed,
            print_filter_warnings=print_filter_warnings,
        )
        return iterator
        
    # Create iterators for short and long sequences
    iterators = []
    for T in [short_T, long_T]:
        # Create a modified dataloader config with the specific T
        dl_cfg = DataloaderConfig(
            B=dataloader_cfg.B,
            T=T,
            num_workers=num_workers // 2,
            prefetch_buffer_size=(prefetch_buffer_size + 1) // 2,
            device_prefetch_buffer_size=(device_prefetch_buffer_size + 1) // 2,
            short_T=short_T,
            long_T=long_T,
            long_ratio=long_ratio,
            start_step=dataloader_cfg.start_step,
        )
        it = make_iterator(
            cfg=dl_cfg,
            seed=seed + T,
            print_filter_warnings=print_filter_warnings,
            device=device,
        )
        iterators.append(it)

    iterator = AlternatingIterator(iterators[0], iterators[1], long_ratio, seed)
    return iterator
    
