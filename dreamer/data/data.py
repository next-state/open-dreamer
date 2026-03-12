import logging

import grain
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
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
    dataloader_cfg: DataloaderConfig | None = None,
    seq_len: int | None = None,
    pack_factor: int = 1,
):
    """Creates a data loading pipeline using Grain from a DatasetConfig.

    Args:
        cfg: Dataset configuration
        dataloader_cfg: Optional dataloader override (defaults to cfg.dataloader_cfg)
        seq_len: Optional explicit sequence length override
        seed: Random seed
        print_filter_warnings: Whether to print filter warnings
        device: Device for prefetching
        pack_factor: Number of episodes to pack into one sequence. When >1, batches
            pack_factor * B episodes of length seq_len and reshapes to B sequences
            of length seq_len * pack_factor.
    """
    if dataloader_cfg is None:
        dataloader_cfg = cfg.dataloader_cfg

    if seq_len is None:
        seq_len = dataloader_cfg.long_T
    
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
        num_records = cfg.num_max_samples if cfg.num_max_samples>0 else len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    # Build operations based on dataset type and data type
    if use_latent_data:
        # Pre-tokenized latent data path
        operations = [ProcessLatentAndSlice(seq_len=seq_len)]
    elif cfg.name.startswith("minecraft_vpt"):
        operations = [
            EpisodeLengthFilter(
                seq_len=seq_len,
                format_hint="vpt",
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessMinecraftEpisodeAndSlice(
                seq_len=seq_len,
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
                seq_len=seq_len,
                format_hint="coinrun",
                print_filter_warnings=print_filter_warnings,
            ),
            ProcessEpisodeAndSlice(
                seq_len=seq_len,
                image_h=cfg.H,
                image_w=cfg.W,
                image_c=cfg.C,
                padding_h=cfg.padding_H,
                padding_w=cfg.padding_W,
                p_include_reward=cfg.p_include_reward,
                patch_size=cfg.patch_size,
            ),
        ]
            
    batch_size = per_process_batch_size * pack_factor
    common_ops = [Batch(batch_size=batch_size, drop_remainder=True)]
    if pack_factor > 1:
        common_ops.append(PackEpisodes(pack_factor))
    common_ops.append(CastDtype(dataloader_cfg.dtype))
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


# ==============================================================================
# Episode Packing
# ==============================================================================

class PackEpisodes(grain.transforms.Map):
    """Reshape a batch of short episodes into packed long sequences.

    Transforms (B * n_splits, short_T, ...) → (B, n_splits * short_T, ...)
    so that each packed sequence contains n_splits independent episodes.
    """

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def map(self, batch):
        def _reshape(x):
            if x is None:
                return None
            shape = x.shape
            B_packed = shape[0] // self.n_splits
            new_shape = (B_packed, self.n_splits * shape[1], *shape[2:])
            return x.reshape(new_shape)
        return jax.tree.map(_reshape, batch)


# ==============================================================================
# Dual Iterator (uniform output shapes)
# ==============================================================================

class AlternatingIterator:
    """Alternates between short (packed) and long iterators with uniform output shapes.

    Both iterators must produce batches of the same shape (B, long_T, ...).
    Adds 'n_splits' to each batch dict to indicate block-causal chunk count.
    """

    def __init__(self, short_iterator, long_iterator, n_splits: int, long_ratio: float, seed: int) -> None:
        self.short_iterator = iter(short_iterator)
        self.long_iterator = iter(long_iterator)
        self.n_splits = n_splits
        self.long_ratio = long_ratio
        self._rng_key = jax.random.key(seed)
        self.step = 0

    def __next__(self):
        self._rng_key, subkey = jax.random.split(self._rng_key)
        use_long = bool(jax.random.bernoulli(subkey, p=self.long_ratio)) or self.step % 1000 == 0
        self.step += 1

        if use_long:
            batch = next(self.long_iterator)
            batch["n_splits"] = 1
        else:
            batch = next(self.short_iterator)
            batch["n_splits"] = self.n_splits
        return batch

    def __iter__(self):
        return self


def make_dual_iterator(
    cfg: DatasetConfig,
    *,
    seed: int = 42,
    print_filter_warnings: bool = False,
    device=None,
    dtype=None,
) -> AlternatingIterator:
    """Create alternating iterator over short (packed) and long sequences.

    Both pipelines output the same shape (B, long_T, ...) for uniform GPU utilization.
    Short sequences are packed from n_splits independent episodes with a block-causal mask.

    Args:
        cfg: Dataset configuration
        seed: Random seed
        print_filter_warnings: Whether to print filter warnings
        device: Device for prefetching
        dtype: Optional dtype override
    """
    dataloader_cfg = cfg.dataloader_cfg

    short_T = dataloader_cfg.short_T
    long_T = dataloader_cfg.long_T
    assert long_T % short_T == 0, f"long_T ({long_T}) must be a multiple of short_T ({short_T})"
    n_splits = long_T // short_T
    long_ratio = dataloader_cfg.long_ratio

    if short_T == long_T:
        logging.warning("short_T == long_T, using just one iterator")
        return make_iterator(cfg, seed=seed, print_filter_warnings=print_filter_warnings, device=device, dtype=dtype, seq_len=long_T)

    # Long pipeline: single episodes of length long_T, batch B → (B, long_T, ...)
    long_iter = make_iterator(
        cfg, seed=seed, print_filter_warnings=print_filter_warnings,
        device=device, dtype=dtype, seq_len=long_T,
    )

    # Short pipeline: short_T episodes, batch B*n_splits, pack → (B, long_T, ...)
    short_iter = make_iterator(
        cfg, seed=seed + short_T, print_filter_warnings=print_filter_warnings,
        device=device, dtype=dtype, seq_len=short_T, pack_factor=n_splits,
    )

    return AlternatingIterator(short_iter, long_iter, n_splits, long_ratio, seed)


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
  
  def _make_sharding(leaf):
    """Adapt sharding to leaf rank by truncating the partition spec."""
    if not isinstance(device, jax.sharding.NamedSharding):
      return device
    if not hasattr(leaf, 'ndim'):
      return None
    if leaf.ndim >= len(device.spec):
      return device
    truncated = jax.sharding.PartitionSpec(*device.spec[:leaf.ndim])
    return jax.sharding.NamedSharding(device.mesh, truncated)

  def _transfer(x):
    def _put_leaf(leaf):
      sharding = _make_sharding(leaf)
      if sharding is None:
        return leaf
      if isinstance(sharding, jax.sharding.NamedSharding) and not sharding.is_fully_addressable:
        # Multi-host input path: convert process-local batches to global arrays.
        return multihost_utils.host_local_array_to_global_array(leaf, sharding.mesh, sharding.spec)
      return jax.device_put(leaf, sharding)

    x = jax.tree.map(_put_leaf, x)
    if jax_dtype is not None:
      x = jax.tree.map(lambda a: a.astype(jax_dtype) if a is not None and jnp.issubdtype(a.dtype, jnp.floating) else a, x)
    return x
    
  ds = ds.map(_transfer)
  ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=device_buffer_size)
  return ds
