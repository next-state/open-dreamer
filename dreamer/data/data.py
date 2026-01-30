import copy
import io
import pickle
from typing import Any

import grain
import jax
from jax import numpy as jnp
import numpy as np
from grain._src.python.dataset import dataset as grain_dataset
from grain.transforms import Batch

from dreamer.actions import Actions
from dreamer.configs import DatasetConfig, DataloaderConfig
from dreamer.data.path_utils import build_dataset_paths
from dreamer.data.serialization import deserialize_msgpack_record
from dreamer.data.transforms import (
    CastDtype,
    EpisodeLengthFilter,
    ProcessEpisodeAndSlice,
    ProcessLatentAndSlice,
    ProcessMinecraftEpisodeAndSlice,
)

# ==============================================================================
# Factory
# ==============================================================================

def make_iterator(
    cfg: DatasetConfig,
    *,
    seed: int = 42,
    print_filter_warnings: bool = False,
    device = None,
    dtype = None,
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
            
    common_ops = [Batch(batch_size=per_process_batch_size, drop_remainder=True), CastDtype(dataloader_cfg.dtype)]
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
        dtype=dtype,
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
    dtype = None
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
            device=device,
            dtype=dtype,
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
            start_step=dataloader_cfg.start_step,
            dtype=dataloader_cfg.dtype,
        )
        it = make_iterator(
            cfg=cfg,
            seed=seed + T,
            print_filter_warnings=print_filter_warnings,
            device=device,
            dtype=dtype,
        )
        iterators.append(it)

    iterator = AlternatingIterator(iterators[0], iterators[1], long_ratio, seed)
    return iterator
    


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
# Adapted from grain.experimental.device_put
# ==============================================================================

from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations.prefetch import ThreadPrefetchIterDataset


def device_put(
    ds: dataset.IterDataset,
    device,
    *,
    dtype: str | None = None,
    cpu_buffer_size: int = 4,
    device_buffer_size: int = 2,
) -> dataset.IterDataset:
  """Moves the data to the given devices with prefetching.

  Stage 1: A CPU-side prefetch buffer.
  Stage 2: Per-device buffers for elements already transferred to the device.

  Args:
    ds: Dataset to prefetch.
    device: same arguments as in jax.device_put.
    dtype: Optional dtype to cast floating-point arrays to (e.g., "float32", "bfloat16").
    cpu_buffer_size: Number of elements to prefetch on CPU.
    device_buffer_size: Number of elements to prefetch per device.

  Returns:
    Dataset with the elements prefetched to the devices.
  """
  ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cpu_buffer_size)
  # May raise ImportError if jax is not linked.

  jax_dtype = getattr(jnp, dtype, None) if dtype else None
  
  def _transfer(x):
    x = jax.device_put(x, device)
    if jax_dtype is not None:
      x = jax.tree.map(lambda a: a.astype(jax_dtype) if a is not None and jnp.issubdtype(a.dtype, jnp.floating) else a, x)
    return x
    
  ds = ds.map(_transfer)
  ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=device_buffer_size)
  return ds
