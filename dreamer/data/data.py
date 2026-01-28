import copy

import jax
import numpy as np
import grain
from grain._src.python.dataset import dataset as grain_dataset

from ..configs import DatasetConfig
from .transforms import (
    EpisodeLengthFilter,
    ProcessEpisodeAndSlice,
    ProcessMinecraftEpisodeAndSlice,
    ProcessLatentAndSlice,
    CreateActions,
)
from .path_utils import build_dataset_paths
from grain.experimental import device_put
from grain.transforms import Batch


# ==============================================================================
# DataLoader to IterDataset Wrapper
# ==============================================================================

class DataLoaderIteratorWrapper(grain_dataset.DatasetIterator):
    """Wraps a DataLoader iterator as a DatasetIterator for use with grain.experimental.device_put."""
    
    def __init__(self, dataloader_iterator):
        super().__init__()
        self._iterator = dataloader_iterator
        self._count = 0
    
    def __next__(self):
        self._assert_not_closed()
        self._count += 1
        return next(self._iterator)
    
    def get_state(self):
        # Basic state - just track count for debugging
        # Full checkpointing would need access to underlying dataloader state
        return {"count": self._count}
    
    def set_state(self, state):
        self._count = state.get("count", 0)


class DataLoaderIterDataset(grain_dataset.IterDataset):
    """Wraps a DataLoader as an IterDataset for use with grain.experimental.device_put."""
    
    def __init__(self, dataloader):
        super().__init__()
        self._dataloader = dataloader
    
    def __iter__(self):
        return DataLoaderIteratorWrapper(iter(self._dataloader))


# ==============================================================================
# Factory
# ==============================================================================

def make_iterator(
    cfg: DatasetConfig,
    num_workers: int = 16,
    prefetch_buffer_size: int = 10,
    seed: int = 42,
    print_filter_warnings: bool = False,
    device = None,
    device_prefetch_buffer_size: int = 1
):
    """Creates a data loading pipeline using Grain from a DatasetConfig.

    Args:
        cfg: Dataset configuration
        num_workers: Number of worker processes
        prefetch_buffer_size: Prefetch buffer size
        seed: Random seed
        print_filter_warnings: Whether to print filter warnings
    """
    # Build array record paths using utility
    use_latent_data = cfg.data_type == "latent"
    dataset_type = "latent" if use_latent_data else cfg.name

    array_record_paths = build_dataset_paths(
        cfg.array_record_path,
        dataset_type=dataset_type,
        index_max=cfg.index_max if use_latent_data or cfg.name == "minecraft_vpt" else None,
    )

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
            # EpisodeLengthFilter(
            #     seq_len=cfg.T,
            #     format_hint="latent",
            #     print_filter_warnings=print_filter_warnings,
            # ),
            ProcessLatentAndSlice(seq_len=cfg.T),
            Batch(batch_size=per_process_batch_size, drop_remainder=True),
            CreateActions()
        ]
    elif cfg.name == "minecraft_vpt":
        operations = [
            EpisodeLengthFilter(
                seq_len=cfg.T,
                format_hint="vpt",
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessMinecraftEpisodeAndSlice(
                seq_len=cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                patch_size=cfg.patch_size,
            ),
            Batch(batch_size=per_process_batch_size, drop_remainder=True),
            CreateActions(),
        ]
    else:
        operations = [
            EpisodeLengthFilter(
                seq_len=cfg.T,
                format_hint="coinrun",
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessEpisodeAndSlice(
                seq_len=cfg.T,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                p_include_reward=cfg.p_include_reward,
                patch_size=cfg.patch_size,
            ),
            Batch(batch_size=per_process_batch_size, drop_remainder=True),
            CreateActions(),
        ]

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
    iter_dataset = DataLoaderIterDataset(dataloader)
    iter_dataset = device_put(
        iter_dataset,
        device,
        cpu_buffer_size=prefetch_buffer_size,
        device_buffer_size=device_prefetch_buffer_size if prefetch_buffer_size is not None else prefetch_buffer_size,
    )
    
    return iter_dataset

class AlternatingIterator:
    def __init__(
        self,
        short_iterator,
        long_iterator,
        long_ratio: float,
        *,
        start_step: int = 0,
    ) -> None:
        self.short_iterator = short_iterator
        self.long_iterator = long_iterator
        self.long_ratio = long_ratio

        if long_ratio <= 0.0:
            self._mode = "short"
            self._long_budget = 0.0
        elif long_ratio >= 1.0:
            self._mode = "long"
            self._long_budget = 0.0
        else:
            self._mode = "mixed"
            self._long_budget = (start_step * long_ratio) % 1.0

    def __iter__(self):
        return self

    def __next__(self):
        if self._mode == "short":
            return False, next(self.short_iterator)
        if self._mode == "long":
            return True, next(self.long_iterator)

        self._long_budget += self.long_ratio
        if self._long_budget >= 1.0:
            self._long_budget -= 1.0
            return True, next(self.long_iterator)
        return False, next(self.short_iterator)

    @property
    def iterators(self) -> dict[str, grain.DataLoaderIterator]:
        return {
            "short_dataloader_state": self.short_iterator,
            "long_dataloader_state": self.long_iterator,
        }


def make_dual_iterator(
    cfg: DatasetConfig,
    short_T: int,
    long_T: int,
    long_ratio: float,
    *,
    start_step: int = 0,
    iterators: dict[str, grain.DataLoaderIterator] | None = None,
    num_workers: int = 22,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
    print_filter_warnings: bool = False,
) -> AlternatingIterator:
    """Create alternating iterator over short and long sequences."""
    if iterators is None:
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

        short_iterator = iter(short_loader)
        long_iterator = iter(long_loader)
    else:
        short_iterator = iterators["short_dataloader_state"]
        long_iterator = iterators["long_dataloader_state"]

    return AlternatingIterator(
        short_iterator,
        long_iterator,
        long_ratio,
        start_step=start_step,
    )
