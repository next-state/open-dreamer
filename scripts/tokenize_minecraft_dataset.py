"""
Tokenize Minecraft VPT dataset.

Reads existing ArrayRecord shards containing MP4-encoded Minecraft VPT videos,
decodes them, encodes them into latents using a pretrained tokenizer, and writes
new ArrayRecord shards with latents (using msgpack serialization).

3-stage async pipeline:
  [Grain DataLoader] → [CPU Prefetch] → [Device Transfer + Prefetch] → [Main Loop] → [Async Writer]

Example usage:
    python scripts/tokenize_minecraft_dataset.py \
        tokenizer_ckpt=/path/to/tokenizer/ckpt \
        output_dir=/home/ubuntu/minecraft-vpt/arrayrecords-latents
"""
import logging
import os
import queue
from pathlib import Path
from threading import Thread

import decord
import grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from grain._src.python.dataset import dataset as grain_dataset
from grain._src.python.dataset.transformations.prefetch import (
    ThreadPrefetchIterDataset,
)
from omegaconf import DictConfig
from tqdm import tqdm

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.data.data import DataLoaderIteratorWrapper
from dreamer.data.shard_writer import ShardWriter
from dreamer.data.transforms import ProcessMinecraftEpisodeAndSlice
from dreamer.parallel import build_parallel

decord.bridge.set_bridge("native")

logging.getLogger("absl").setLevel(logging.WARNING)
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'


# ==============================================================================
# Data Loading
# ==============================================================================


def make_tokenization_iterator(
    input_dir: str,
    index_max: int | None,
    batch_size: int,
    num_workers: int,
    *,
    image_h: int,
    image_w: int,
    image_c: int,
    padding_h: tuple[int, int],
    padding_w: tuple[int, int],
    patch_size: int,
    mouse_repr: str = "categorical",
):
    """Create dataloader for tokenization (sequential, no shuffle, full episodes)."""
    all_shards = sorted(Path(input_dir).glob("shard-*.array_record"))
    if index_max is not None:
        all_shards = all_shards[:index_max]
    shard_paths = [str(p) for p in all_shards]

    if not shard_paths:
        raise ValueError(f"No shards found in {input_dir}")

    print(f"[tokenize] Found {len(shard_paths)} shards")

    source = grain.sources.ArrayRecordDataSource(shard_paths)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
    )

    operations = [
        ProcessMinecraftEpisodeAndSlice(
            seq_len=0,
            image_h=image_h,
            image_w=image_w,
            image_c=image_c,
            padding_h=padding_h,
            padding_w=padding_w,
            patch_size=patch_size,
            full_episode=True,
            mouse_repr=mouse_repr,
        ),
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=1,
    )


# ==============================================================================
# Async Shard Writer
# ==============================================================================


class AsyncShardWriter:
    """Wraps ShardWriter with a background thread so disk I/O never blocks the main loop."""

    def __init__(
        self,
        output_dir: Path | str,
        records_per_shard: int = 5000,
        maxsize: int = 100,
    ):
        self._writer = ShardWriter(output_dir, records_per_shard)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self):
        for item in iter(self._queue.get, None):
            self._writer.write(item)

    def write(self, record: dict):
        self._queue.put(record)

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self._writer.close()

    @property
    def shard_idx(self):
        return self._writer.shard_idx

    @property
    def total_records(self):
        return self._writer._total_records


# ==============================================================================
# Device-sharded prefetch pipeline
# ==============================================================================


def build_prefetch_pipeline(
    dataloader,
    sharding: jax.sharding.NamedSharding,
    cpu_buffer_size: int = 10,
    device_buffer_size: int = 2,
):
    """Build a 2-stage async prefetch pipeline: CPU buffer → device transfer + device buffer.

    Returns an iterable that yields batches with 'videos' already on device as sharded JAX arrays,
    plus '_pad_size' and '_batch_size' metadata.
    """
    devices = list(sharding.mesh.devices.flat)
    num_devices = len(devices)

    def transfer_to_devices(batch):
        videos = batch["videos"]  # (B, T, H, W, C) numpy float32
        batch_size = videos.shape[0]
        pad_size = (num_devices - batch_size % num_devices) % num_devices

        if pad_size > 0:
            padding = np.zeros((pad_size,) + videos.shape[1:], dtype=videos.dtype)
            videos = np.concatenate([videos, padding], axis=0)

        # Cast to bfloat16 on device to halve HBM usage
        per_device = videos.shape[0] // num_devices
        shards = [
            jax.device_put(videos[i * per_device : (i + 1) * per_device], devices[i]).astype(jnp.bfloat16)
            for i in range(num_devices)
        ]
        shape = videos.shape
        sharded = jax.make_array_from_single_device_arrays(
            shape, sharding, shards
        )

        batch["videos"] = sharded
        batch["_pad_size"] = pad_size
        batch["_batch_size"] = batch_size
        return batch

    # Stage 1: CPU-side prefetch buffer (decouples grain workers from main loop)
    iter_ds = DataLoaderIteratorWrapper(dataloader)
    iter_ds = ThreadPrefetchIterDataset(iter_ds, prefetch_buffer_size=cpu_buffer_size)

    # Stage 2: Device transfer + device-side buffer
    iter_ds = iter_ds.map(transfer_to_devices)
    iter_ds = ThreadPrefetchIterDataset(
        iter_ds, prefetch_buffer_size=device_buffer_size
    )

    return iter_ds


# ==============================================================================
# Vectorized Welford stats
# ==============================================================================


class WelfordAccumulator:
    """Batch-vectorized Welford online statistics (O(1) numpy ops per batch)."""

    def __init__(self, n_channels: int):
        self.count = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.m2 = np.zeros(n_channels, dtype=np.float64)

    def update(self, flat: np.ndarray):
        """Update with a batch of samples. flat: (N, D) float array."""
        batch_count = flat.shape[0]
        batch_mean = flat.mean(axis=0).astype(np.float64)
        batch_var = flat.var(axis=0).astype(np.float64)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean += delta * batch_count / total_count
        self.m2 += (
            batch_var * batch_count
            + delta**2 * self.count * batch_count / total_count
        )
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros_like(self.mean)
        return np.sqrt(self.m2 / self.count)


# ==============================================================================
# Main
# ==============================================================================


def run(cfg: DictConfig):
    print(f"[tokenize] Loading tokenizer from: {cfg.tokenizer_ckpt}")
    print(f"[tokenize] Input directory: {cfg.dataset.array_record_path}")
    print(f"[tokenize] Output directory: {cfg.output_dir}")
    dataloader_cfg = cfg.dataset.dataloader_cfg

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        # Load pretrained tokenizer
        bundle = TokenizerCheckpointBundle.from_pretrained(
            cfg.tokenizer_ckpt,
            mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer
        del tokenizer.decoder  # Free decoder params — only encoding

        print(f"[tokenize] Tokenizer loaded successfully")
        print(f"[tokenize] n_latents: {tokenizer.encoder.n_latents}")
        print(f"[tokenize] d_bottleneck: {tokenizer.cfg.encoder.d_bottleneck}")
        print(f"[tokenize] patch_size: {tokenizer.cfg.encoder.patch_size}")

        # Create dataloader
        base_dataloader = make_tokenization_iterator(
            input_dir=cfg.dataset.array_record_path,
            index_max=cfg.dataset.index_max,
            batch_size=dataloader_cfg.B,
            num_workers=dataloader_cfg.num_workers,
            image_h=cfg.dataset.H,
            image_w=cfg.dataset.W,
            image_c=cfg.dataset.C,
            padding_h=cfg.dataset.padding_H,
            padding_w=cfg.dataset.padding_W,
            patch_size=cfg.dataset.patch_size,
            mouse_repr=cfg.dataset.mouse_repr,
        )

        # Build async prefetch pipeline: CPU buffer → device transfer → device buffer
        prefetched = build_prefetch_pipeline(
            base_dataloader,
            sharding=data_sharding,
            cpu_buffer_size=dataloader_cfg.prefetch_buffer_size,
            device_buffer_size=dataloader_cfg.device_prefetch_buffer_size,
        )

        # Create async shard writer (disk I/O on background thread)
        output_dir = Path(cfg.output_dir)
        writer = AsyncShardWriter(
            output_dir,
            records_per_shard=cfg.records_per_shard,
        )

        # Create metadata directory for stats
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        stats_path = metadata_dir / "latent_stats.npz"

        # JIT-compile encode function
        @nnx.jit
        def encode_batch(videos):
            latents, _ = tokenizer.encode(
                videos,
                deterministic=True,
                rngs=nnx.Rngs(0),
            )
            return latents

        # Vectorized Welford accumulator
        n_channels = tokenizer.cfg.encoder.d_bottleneck
        welford = WelfordAccumulator(n_channels)
        total_videos = 0

        import time

        try:
            pbar = tqdm(prefetched, desc="Tokenizing")
            t_iter_start = time.perf_counter()
            for batch in pbar:
                t_iter = time.perf_counter() - t_iter_start

                videos = batch["videos"]  # Already sharded JAX array
                actions_batch = batch["actions"]
                sources_batch = batch["source"]
                batch_size = batch["_batch_size"]
                pad_size = batch["_pad_size"]

                t0 = time.perf_counter()
                latents = encode_batch(videos)
                latents.block_until_ready()
                t1 = time.perf_counter()

                latents_np = np.asarray(latents)[:batch_size]
                t2 = time.perf_counter()

                # Vectorized Welford update: flatten (B, T, n_latents) → (N, D)
                welford.update(latents_np.astype(np.float32).reshape(-1, n_channels)) # Cast to float32 first — bfloat16 var() has catastrophic cancellation
                t3 = time.perf_counter()

                # Queue records for async write
                for i in range(batch_size):
                    actions_i = { k: v[i] if v is not None else None for k, v in actions_batch.items()}
                    writer.write(
                        {
                            "latents": latents_np[i],  # (T, n_latents, d_bottleneck)
                            "actions": actions_i,
                            "source": (
                                sources_batch[i]
                                if sources_batch is not None
                                else None
                            ),
                        }
                    )
                t4 = time.perf_counter()

                pbar.set_postfix(
                    mean=f"{welford.mean.mean():.4f}",
                    std=f"{welford.std.mean():.4f}",
                    fetch=f"{t_iter:.3f}s",
                    encode=f"{t1-t0:.3f}s",
                    d2h=f"{t2-t1:.3f}s",
                    welford=f"{t3-t2:.3f}s",
                    queue=f"{t4-t3:.3f}s",
                )

                total_videos += batch_size
                t_iter_start = time.perf_counter()

                # Periodically save stats (every 100 batches)
                if total_videos % (100 * dataloader_cfg.B) < dataloader_cfg.B:
                    np.savez(
                        stats_path,
                        mean=welford.mean.astype(np.float32),
                        std=welford.std.astype(np.float32),
                        num_samples=welford.count,
                        num_videos=total_videos,
                    )

        finally:
            writer.close()
            # Save final stats
            if welford.count > 0:
                np.savez(
                    stats_path,
                    mean=welford.mean.astype(np.float32),
                    std=welford.std.astype(np.float32),
                    num_samples=welford.count,
                    num_videos=total_videos,
                )

        print(f"[tokenize] Done! Processed {total_videos} videos")
        print(f"[tokenize] Wrote {writer.shard_idx} shards to {cfg.output_dir}")
        print(f"[tokenize] Total records: {writer.total_records}")
        print(f"[tokenize] Latent stats saved to: {stats_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="tokenize")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
