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
import pickle
import queue
from pathlib import Path
from threading import Thread

import decord
import grain
import hydra
import jax
import numpy as np
from flax import nnx
from grain._src.python.dataset import dataset as grain_dataset
from grain._src.python.dataset.transformations.prefetch import (
    ThreadPrefetchIterDataset,
)
from omegaconf import DictConfig
from tqdm import tqdm

from dreamer.actions import parse_action_dicts
from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.data.data import DataLoaderIteratorWrapper
from dreamer.data.shard_writer import ShardWriter
from dreamer.data.transforms import ProcessMinecraftEpisodeAndSlice
from dreamer.parallel import build_parallel

decord.bridge.set_bridge("native")

logging.getLogger("absl").setLevel(logging.WARNING)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# ==============================================================================
# Data Loading
# ==============================================================================


class MinecraftVPTProcessFullEpisode(grain.transforms.Map):
    """Decode full MP4 video for tokenization (no slicing).

    Uses ProcessMinecraftEpisodeAndSlice with full_episode=True, then adds action parsing.
    """

    def __init__(self, patch_size: int = 16):
        self.patch_size = patch_size
        self.processor = ProcessMinecraftEpisodeAndSlice(
            seq_len=0,
            image_h=128,
            image_w=128,
            image_c=3,
            full_episode=True,
        )

    def _calculate_padding(self, dimension: int) -> tuple[int, int]:
        """Calculate (top/left, bottom/right) padding to make dimension divisible by patch_size."""
        if dimension % self.patch_size == 0:
            return (0, 0)
        padding_needed = self.patch_size - (dimension % self.patch_size)
        padding_start = padding_needed // 2
        padding_end = padding_needed - padding_start
        return (padding_start, padding_end)

    def map(self, element: bytes) -> dict:
        rng = np.random.default_rng(0)
        result = self.processor.random_map(element, rng)

        data = pickle.loads(element)

        frames = result["videos"]  # (T, H, W, C) in [0, 255]
        padding_h = self._calculate_padding(frames.shape[1])
        padding_w = self._calculate_padding(frames.shape[2])
        frames = np.pad(
            frames,
            ((0, 0), padding_h, padding_w, (0, 0)),
            mode="constant",
            constant_values=0,
        )

        actions = data.get("actions")
        actions = parse_action_dicts(actions).to_dict()

        return {
            "videos": frames,  # (T, H_padded, W_padded, C) in [0, 255]
            "actions": actions,
            "source": data.get("source"),
        }


def make_tokenization_iterator(
    input_dir: str,
    num_shards: int | None,
    batch_size: int,
    num_workers: int,
    patch_size: int = 16,
):
    """Create dataloader for tokenization (sequential, no shuffle, full episodes)."""
    all_shards = sorted(Path(input_dir).glob("shard-*.array_record"))
    if num_shards is not None:
        all_shards = all_shards[:num_shards]
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
        MinecraftVPTProcessFullEpisode(patch_size=patch_size),
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
        records_per_shard: int = 1000,
        serialization_format: str = "msgpack",
        maxsize: int = 100,
    ):
        self._writer = ShardWriter(output_dir, records_per_shard, serialization_format)
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
        videos = batch["videos"]  # (B, T, H, W, C) numpy
        batch_size = videos.shape[0]
        pad_size = (num_devices - batch_size % num_devices) % num_devices

        if pad_size > 0:
            padding = np.zeros((pad_size,) + videos.shape[1:], dtype=videos.dtype)
            videos = np.concatenate([videos, padding], axis=0)

        per_device = videos.shape[0] // num_devices
        shards = [
            jax.device_put(videos[i * per_device : (i + 1) * per_device], devices[i])
            for i in range(num_devices)
        ]
        sharded = jax.make_array_from_single_device_arrays(
            videos.shape, sharding, shards
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

    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    with jax.set_mesh(mesh):
        # Load pretrained tokenizer
        bundle = TokenizerCheckpointBundle.from_pretrained(
            cfg.tokenizer_ckpt,
            mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer

        print(f"[tokenize] Tokenizer loaded successfully")
        print(f"[tokenize] n_latents: {tokenizer.encoder.n_latents}")
        print(f"[tokenize] d_bottleneck: {tokenizer.cfg.encoder.d_bottleneck}")
        print(f"[tokenize] patch_size: {tokenizer.cfg.encoder.patch_size}")

        # Create dataloader
        base_dataloader = make_tokenization_iterator(
            input_dir=cfg.dataset.array_record_path,
            num_shards=cfg.num_shards,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            patch_size=tokenizer.cfg.encoder.patch_size,
        )

        # Build async prefetch pipeline: CPU buffer → device transfer → device buffer
        prefetched = build_prefetch_pipeline(
            base_dataloader,
            sharding=data_sharding,
            cpu_buffer_size=10,
            device_buffer_size=2,
        )

        # Create async shard writer (disk I/O on background thread)
        output_dir = Path(cfg.output_dir)
        writer = AsyncShardWriter(
            output_dir,
            records_per_shard=cfg.records_per_shard,
            serialization_format="msgpack",
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

        try:
            pbar = tqdm(prefetched, desc="Tokenizing")
            for batch in pbar:
                videos = batch["videos"]  # Already sharded JAX array
                actions_batch = batch["actions"]
                sources_batch = batch["source"]
                batch_size = batch["_batch_size"]
                pad_size = batch["_pad_size"]

                latents = encode_batch(videos)

                # Gather latents from each device shard separately (avoids GPU 0 memory spike)
                latents_np = np.concatenate(
                    [np.asarray(shard.data) for shard in latents.addressable_shards],
                    axis=0,
                )

                # Remove padding
                if pad_size > 0:
                    latents_np = latents_np[:batch_size]

                # Vectorized Welford update: flatten (B, T, n_latents) → (N, D)
                welford.update(latents_np.reshape(-1, n_channels))

                pbar.set_postfix(
                    mean=f"{float(welford.mean.mean()):.4f}",
                    std=f"{float(welford.std.mean()):.4f}",
                )

                # Queue records for async write
                for i in range(batch_size):
                    actions_i = {
                        k: v[i] if v is not None else None
                        for k, v in actions_batch.items()
                    }
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

                total_videos += batch_size

                # Periodically save stats (every 100 batches)
                if total_videos % (100 * cfg.batch_size) < cfg.batch_size:
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
