"""
Tokenize Minecraft VPT dataset.

Reads existing ArrayRecord shards containing MP4-encoded Minecraft VPT videos,
decodes them, encodes them into latents using a pretrained tokenizer, and writes
new ArrayRecord shards with latents (using msgpack serialization).

Example usage:
    python scripts/tokenize_minecraft_dataset.py \
        --tokenizer_ckpt /path/to/tokenizer/ckpt \
        --input_dir /home/ubuntu/minecraft-vpt/arrayrecords-mp4 \
        --output_dir /home/ubuntu/minecraft-vpt/arrayrecords-latents \
        --batch_size 8
"""
import argparse
import io
import logging
import os
import pickle
import queue
import threading
from pathlib import Path
from typing import Any

import decord
import grain
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from dreamer.actions import parse_action_dicts
from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.parallel import build_parallel
from dreamer.data.shard_writer import ShardWriter

decord.bridge.set_bridge("native")

# Suppress absl info logs
logging.getLogger("absl").setLevel(logging.WARNING)

# Disable JAX preallocation
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# ==============================================================================
# Serialization (imported from shared module)
# ==============================================================================

from dreamer.data.serialization import serialize_msgpack_record, deserialize_msgpack_record

# Backward compatibility aliases
serialize_record = serialize_msgpack_record
deserialize_record = deserialize_msgpack_record
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

VIDEO_DTYPE_MAP = {
    "uint8": np.uint8,
    "float32": np.float32,
}

LATENT_DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
}


# ==============================================================================
# Data Loading
# ==============================================================================


class MinecraftVPTProcessFullEpisode(grain.transforms.Map):
    """Decode full MP4 episodes for tokenization and parse VPT actions.

    When image_h and image_w are provided, decord decodes directly at that
    resolution (reduces memory vs decoding full-res then resizing).
    """

    def __init__(
        self,
        *,
        video_dtype: np.dtype,
        decord_threads: int,
        image_h: int | None = None,
        image_w: int | None = None,
    ):
        self.video_dtype = np.dtype(video_dtype)
        self.decord_threads = max(1, int(decord_threads))
        self.image_h = image_h
        self.image_w = image_w

    def map(self, element: bytes) -> dict[str, Any]:
        # Parse once (the previous implementation parsed twice per sample).
        data = pickle.loads(element)

        # Decode at target resolution when image_h/image_w set (reduces memory vs decode-then-resize)
        mp4_bytes = io.BytesIO(data["video"])
        vr_kw = {"ctx": decord.cpu(0), "num_threads": self.decord_threads}
        if self.image_h is not None and self.image_w is not None:
            vr_kw["width"] = self.image_w
            vr_kw["height"] = self.image_h
        vr = decord.VideoReader(mp4_bytes, **vr_kw)

        frame_indices = np.arange(len(vr), dtype=np.int64)
        videos = vr.get_batch(frame_indices).asnumpy()
        if videos.dtype != self.video_dtype:
            videos = videos.astype(self.video_dtype, copy=False)

        raw_actions = data.get("actions")
        if raw_actions is None:
            actions = {"binary": None, "categorical": None, "continuous": None}
        else:
            actions = parse_action_dicts(raw_actions).to_dict()

        return {
            "videos": videos,  # (T, H, W, C), typically uint8 in [0, 255]
            "actions": actions,
            "source": data.get("source"),
        }


def make_tokenization_iterator(
    input_dir: str,
    num_shards: int | None,
    batch_size: int,
    num_workers: int,
    worker_buffer_size: int,
    prefetch_buffer_size: int,
    decord_threads: int,
    video_dtype: np.dtype,
    image_h: int | None = None,
    image_w: int | None = None,
):
    """Create dataloader for tokenization (sequential, no shuffle, full episodes).

    When image_h and image_w are provided, decoded frames are resized to that
    resolution before batching (e.g. to match a tokenizer trained at 160x90).
    """
    # Generate shard paths (same pattern as data.py:179)
    all_shards = sorted(Path(input_dir).glob("shard-*.array_record"))
    if num_shards is not None:
        all_shards = all_shards[:num_shards]
    shard_paths = [str(p) for p in all_shards]

    if not shard_paths:
        raise ValueError(f"No shards found in {input_dir}")

    print(f"[tokenize] Found {len(shard_paths)} shards")

    source = grain.sources.ArrayRecordDataSource(shard_paths)

    # Sequential sampler (no shuffle - process once)
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.NoSharding(),  # Single process
        shuffle=False,  # Sequential for tokenization
        num_epochs=1,  # Process once
    )

    operations = [
        MinecraftVPTProcessFullEpisode(
            video_dtype=video_dtype,
            decord_threads=decord_threads,
            image_h=image_h,
            image_w=image_w,
        ),
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=False),
    ]

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=worker_buffer_size,
        read_options=grain.ReadOptions(
            prefetch_buffer_size=prefetch_buffer_size,
            num_threads=1,
        ),
    )


class DeviceShardedIterator:
    """Wraps a dataloader to yield device-sharded JAX arrays.

    Handles padding and direct per-device transfers to avoid GPU 0 memory spikes.
    """

    def __init__(
        self,
        dataloader,
        sharding: jax.sharding.NamedSharding,
        sharded_keys: tuple[str, ...] = ("videos",),
        prefetch_batches: int = 2,
    ):
        self.dataloader = dataloader
        self.sharding = sharding
        self.sharded_keys = sharded_keys
        self.num_devices = len(sharding.mesh.devices.flat)
        self.prefetch_batches = max(1, int(prefetch_batches))

    def _shard_array(self, array: np.ndarray) -> tuple[jax.Array, int]:
        """Shard array across devices, returning (sharded_array, pad_size)."""
        batch_size = array.shape[0]
        pad_size = (self.num_devices - batch_size % self.num_devices) % self.num_devices

        if pad_size > 0:
            pad_spec = [(0, 0)] * array.ndim
            pad_spec[0] = (0, pad_size)
            array = np.pad(array, pad_spec, mode="constant")

        # One sharded transfer lets JAX place each slice on the correct device.
        sharded = jax.device_put(array, self.sharding)
        return sharded, pad_size

    def _iter_batches(self):
        for batch in self.dataloader:
            pad_size = 0
            sharded_batch = {}

            for key, value in batch.items():
                if key in self.sharded_keys:
                    sharded_batch[key], pad_size = self._shard_array(value)
                else:
                    sharded_batch[key] = value

            sharded_batch["_pad_size"] = pad_size
            sharded_batch["_batch_size"] = batch[self.sharded_keys[0]].shape[0]
            yield sharded_batch

    def __iter__(self):
        if self.prefetch_batches <= 1:
            yield from self._iter_batches()
            return

        batch_queue: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_batches)
        sentinel = object()
        errors: list[BaseException] = []

        def _producer():
            try:
                for item in self._iter_batches():
                    batch_queue.put(item)
            except BaseException as exc:  # propagate iterator errors to main thread
                errors.append(exc)
            finally:
                batch_queue.put(sentinel)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        while True:
            item = batch_queue.get()
            if item is sentinel:
                break
            yield item

        producer.join()
        if errors:
            raise errors[0]


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
    default_local_devices = max(1, jax.local_device_count())
    default_batch_size = max(8, 2 * default_local_devices)
    default_num_workers = min(32, max(8, os.cpu_count() or 8))

    parser = argparse.ArgumentParser(
        description="Tokenize Minecraft VPT dataset into latent ArrayRecords"
    )
    parser.add_argument(
        "--tokenizer_ckpt",
        type=str,
        required=True,
        help="Path to pretrained tokenizer checkpoint",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.environ.get(
            "MINECRAFT_VPT_MP4_ARRAYRECORD_DIR",
            "/datasets/minecraft_vpt/arrayrecords-mp4",
        ),
        help="Input shards directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for latent shards",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards to process (for testing, default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_batch_size,
        help="Videos to encode per batch (default: 2x local GPU count)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_num_workers,
        help="Grain dataloader workers",
    )
    parser.add_argument(
        "--worker_buffer_size",
        type=int,
        default=4,
        help="DataLoader worker buffer size",
    )
    parser.add_argument(
        "--prefetch_buffer_size",
        type=int,
        default=8,
        help="ArrayRecord read prefetch depth",
    )
    parser.add_argument(
        "--device_prefetch_batches",
        type=int,
        default=2,
        help="Batches prefetched to device-sharded arrays",
    )
    parser.add_argument(
        "--decord_threads",
        type=int,
        default=2,
        help="Decode threads per sample worker",
    )
    parser.add_argument(
        "--video_dtype",
        type=str,
        default="uint8",
        choices=tuple(VIDEO_DTYPE_MAP.keys()),
        help="Video dtype before transfer to devices",
    )
    parser.add_argument(
        "--latent_dtype",
        type=str,
        default="float32",
        choices=tuple(LATENT_DTYPE_MAP.keys()),
        help="Output latent dtype written to shards",
    )
    parser.add_argument(
        "--stats_save_every",
        type=int,
        default=100,
        help="Save running latent stats every N batches",
    )
    parser.add_argument(
        "--records_per_shard",
        type=int,
        default=1000,
        help="Number of records per output shard",
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=256,
        help="Frames per chunk when encoding long episodes (reduces memory).",
    )
    return parser.parse_args()


def update_welford_batch(
    samples: np.ndarray,
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge a batch into running Welford stats."""
    if samples.size == 0:
        return count, mean, m2

    batch_count = samples.shape[0]
    batch_mean = np.mean(samples, axis=0, dtype=np.float64)
    centered = samples - batch_mean
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)

    if count == 0:
        return batch_count, batch_mean, batch_m2

    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + (delta * delta) * (count * batch_count / total)
    return total, merged_mean, merged_m2


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    video_dtype = np.dtype(VIDEO_DTYPE_MAP[args.video_dtype])
    latent_dtype = np.dtype(LATENT_DTYPE_MAP[args.latent_dtype])

    local_devices = jax.local_device_count()
    global_devices = jax.device_count()
    if args.batch_size < local_devices:
        print(
            f"[tokenize] batch_size {args.batch_size} < local device count "
            f"{local_devices}; bumping batch_size to {local_devices}."
        )
        args.batch_size = local_devices

    print(f"[tokenize] Loading tokenizer from: {args.tokenizer_ckpt}")
    print(f"[tokenize] Input directory: {input_dir}")
    print(f"[tokenize] Output directory: {output_dir}")
    print(f"[tokenize] Device count: local={local_devices}, global={global_devices}")
    print(f"[tokenize] video_dtype={video_dtype}, latent_dtype={latent_dtype}")
    if args.batch_size % local_devices != 0:
        print(
            f"[tokenize] batch_size {args.batch_size} not divisible by "
            f"{local_devices}; padding will be applied on final per-device split."
        )

    # Build parallel setup (data parallelism for tokenization)
    mesh, data_sharding, mesh_rules = build_parallel("data")

    with jax.set_mesh(mesh):
        # Load pretrained tokenizer
        bundle = TokenizerCheckpointBundle.from_pretrained(
            args.tokenizer_ckpt,
            mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer

        print(f"[tokenize] Tokenizer loaded successfully")
        print(f"[tokenize] n_latents: {tokenizer.encoder.n_latents}")
        print(f"[tokenize] d_bottleneck: {tokenizer.cfg.encoder.d_bottleneck}")
        image_h = tokenizer.decoder.H
        image_w = tokenizer.decoder.W
        print(f"[tokenize] Target resolution: H={image_h}, W={image_w}")

        # Create dataloader with device sharding
        base_dataloader = make_tokenization_iterator(
            input_dir=str(input_dir),
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            worker_buffer_size=args.worker_buffer_size,
            prefetch_buffer_size=args.prefetch_buffer_size,
            decord_threads=args.decord_threads,
            video_dtype=video_dtype,
            image_h=image_h,
            image_w=image_w,
        )
        dataloader = DeviceShardedIterator(
            base_dataloader,
            sharding=data_sharding,
            sharded_keys=("videos",),
            prefetch_batches=args.device_prefetch_batches,
        )

        # Create shard writer (using msgpack serialization)
        writer = ShardWriter(
            output_dir,
            records_per_shard=args.records_per_shard,
            serialization_format="msgpack"
        )

        # Create metadata directory for stats
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        stats_path = metadata_dir / "latent_stats.npz"

        # JIT-compile encode function with explicit sharding to ensure SPMD
        # over the data mesh (avoids single-device placement / GPU0 OOM).
        @nnx.jit(
            in_shardings=data_sharding,
            out_shardings=data_sharding,
            donate_argnames=("videos",),
        )
        def encode_batch(videos):
            # Returns unpacked latents: (B, T, n_latents, d_bottleneck)
            latents, _ = tokenizer.encode(
                videos,
                deterministic=True,
                rngs=nnx.Rngs(0),
            )
            return latents

        def _pad_and_shard(chunk_np: np.ndarray, target_len: int) -> jax.Array:
            """Pad a numpy chunk on host then device_put with sharding.

            Keeps the array sharded from the start — no GPU-0 spike.
            """
            pad_t = target_len - chunk_np.shape[1]
            if pad_t > 0:
                pad_spec = [(0, 0)] * chunk_np.ndim
                pad_spec[1] = (0, pad_t)
                chunk_np = np.pad(chunk_np, pad_spec, mode="constant")
            return jax.device_put(chunk_np, data_sharding)

        # Process all batches
        total_videos = 0
        # Welford's online algorithm for numerically stable mean/variance
        # Accumulate per-channel: count, mean, M2 (sum of squared deviations)
        n_channels = tokenizer.cfg.encoder.d_bottleneck
        welford_count = 0
        welford_mean = np.zeros(n_channels, dtype=np.float64)
        welford_m2 = np.zeros(n_channels, dtype=np.float64)
        batch_index = 0

        try:
            pbar = tqdm(dataloader, desc="Tokenizing")
            for batch in pbar:
                batch_index += 1
                videos = batch["videos"]  # Already sharded JAX array
                actions_batch = batch["actions"]
                sources_batch = batch["source"]
                batch_size = batch["_batch_size"]
                pad_size = batch["_pad_size"]

                # Encode in temporal chunks to reduce peak memory.
                # Each chunk is encoded, gathered to numpy immediately, then the
                # JAX array is deleted — so only one chunk lives on device at a
                # time.  Concatenation happens in numpy to avoid GPU-0 gather.
                if args.chunk_len and args.chunk_len > 0:
                    t_total = videos.shape[1]
                    np_chunks = []
                    for start in range(0, t_total, args.chunk_len):
                        end = min(start + args.chunk_len, t_total)
                        chunk_len = end - start

                        if chunk_len < args.chunk_len:
                            # Gather the short tail to numpy, pad there, then
                            # device_put with sharding — avoids recompilation
                            # and keeps the array properly sharded.
                            tail_shards = sorted(
                                videos[:, start:end].addressable_shards,
                                key=lambda s: s.index,
                            )
                            tail_np = np.concatenate(
                                [np.asarray(s.data) for s in tail_shards], axis=0,
                            )
                            chunk = _pad_and_shard(tail_np, args.chunk_len)
                        else:
                            chunk = videos[:, start:end]

                        latents_chunk = encode_batch(chunk)

                        # Immediately gather to numpy per-shard (no GPU-0 spike).
                        chunk_shards = sorted(
                            latents_chunk.addressable_shards,
                            key=lambda s: s.index,
                        )
                        chunk_np = np.concatenate(
                            [np.asarray(s.data) for s in chunk_shards], axis=0,
                        )
                        # Trim temporal padding from the last (short) chunk.
                        if chunk_len < args.chunk_len:
                            chunk_np = chunk_np[:, :chunk_len]
                        np_chunks.append(chunk_np)
                        del latents_chunk  # free device memory before next chunk

                    latents_np = np.concatenate(np_chunks, axis=1)
                else:
                    latents = encode_batch(videos)

                    # Gather from local addressable shards (avoids device-0 gather spikes).
                    shards = sorted(latents.addressable_shards, key=lambda s: s.index)
                    latents_np = np.concatenate(
                        [np.asarray(shard.data) for shard in shards],
                        axis=0,
                    )

                if pad_size > 0:
                    latents_np = latents_np[:batch_size]

                # Vectorized Welford update for this batch.
                latents_flat = latents_np.reshape(-1, n_channels).astype(np.float64, copy=False)
                welford_count, welford_mean, welford_m2 = update_welford_batch(
                    latents_flat,
                    welford_count,
                    welford_mean,
                    welford_m2,
                )
                current_std = (
                    np.sqrt(welford_m2 / welford_count)
                    if welford_count > 0
                    else np.zeros(n_channels, dtype=np.float64)
                )
                pbar.set_postfix(
                    mean=f"{float(welford_mean.mean()):.4f}",
                    std=f"{float(current_std.mean()):.4f}",
                )

                if latents_np.dtype != latent_dtype:
                    latents_np = latents_np.astype(latent_dtype, copy=False)

                binary_actions = actions_batch.get("binary")
                categorical_actions = actions_batch.get("categorical")
                continuous_actions = actions_batch.get("continuous")

                # Write each record individually
                for i in range(batch_size):
                    actions_i = {
                        "binary": binary_actions[i] if binary_actions is not None else None,
                        "categorical": categorical_actions[i] if categorical_actions is not None else None,
                        "continuous": continuous_actions[i] if continuous_actions is not None else None,
                    }
                    record = {
                        "latents": latents_np[i],  # (T, n_latents, d_bottleneck)
                        "actions": actions_i,
                        "source": sources_batch[i] if sources_batch is not None else None,
                    }
                    writer.write(record)

                total_videos += batch_size

                if batch_index % args.stats_save_every == 0 and welford_count > 0:
                    np.savez(
                        stats_path,
                        mean=welford_mean.astype(np.float32),
                        std=current_std.astype(np.float32),
                        num_samples=welford_count,
                        num_videos=total_videos,
                    )

        finally:
            writer.close()
            # Save final stats
            if welford_count > 0:
                final_std = np.sqrt(welford_m2 / welford_count)
                np.savez(
                    stats_path,
                    mean=welford_mean.astype(np.float32),
                    std=final_std.astype(np.float32),
                    num_samples=welford_count,
                    num_videos=total_videos,
                )

        print(f"[tokenize] Done! Processed {total_videos} videos")
        print(f"[tokenize] Wrote {writer.shard_idx} shards to {output_dir}")
        total_records = getattr(writer, "total_records", getattr(writer, "_total_records", None))
        if total_records is not None:
            print(f"[tokenize] Total records: {total_records}")
        print(f"[tokenize] Latent stats saved to: {stats_path}")

        # stats = np.load("output_dir/metadata/latent_stats.npz")                                                                                            
        # print(stats["mean"], stats["std"], stats["num_batches"])    


if __name__ == "__main__":
    main()
