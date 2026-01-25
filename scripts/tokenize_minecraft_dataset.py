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
from pathlib import Path

import decord
import grain
import jax
import msgpack
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
from flax import nnx
from tqdm import tqdm

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.parallel import build_parallel

decord.bridge.set_bridge("native")

# Suppress absl info logs
logging.getLogger("absl").setLevel(logging.WARNING)

# Disable JAX preallocation
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# ==============================================================================
# Serialization
# ==============================================================================


def serialize_record(record: dict) -> bytes:
    """Serialize record using msgpack (no pickle).

    Arrays are serialized via tobytes() with shape/dtype metadata.
    Pre-serialized msgpack bytes (like actions) are stored directly.
    """
    encoded = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            encoded[key] = value.tobytes()
            encoded[f"{key}_shape"] = list(value.shape)
            encoded[f"{key}_dtype"] = str(value.dtype)
        else:
            encoded[key] = value
    return msgpack.packb(encoded, use_bin_type=True)


def deserialize_record(data: bytes) -> dict:
    """Deserialize record from msgpack format.

    Reconstructs numpy arrays from bytes + shape/dtype metadata.
    """
    encoded = msgpack.unpackb(data, raw=False)
    decoded = {}

    # Find all array keys (those with _shape suffix)
    array_keys = {k[:-6] for k in encoded.keys() if k.endswith("_shape")}

    for key, value in encoded.items():
        if key.endswith("_shape") or key.endswith("_dtype"):
            continue
        if key in array_keys:
            shape = tuple(encoded[f"{key}_shape"])
            dtype = encoded[f"{key}_dtype"]
            decoded[key] = np.frombuffer(value, dtype=dtype).reshape(shape)
        else:
            decoded[key] = value

    return decoded


# ==============================================================================
# Data Loading
# ==============================================================================


class MinecraftVPTProcessFullEpisode(grain.transforms.Map):
    """Decode full MP4 video for tokenization (no slicing).

    Reuses the decord pattern from MinecraftVPTProcessEpisodeAndSlice in data.py.
    """

    def map(self, element: bytes) -> dict:
        data = pickle.loads(element)

        # Decord pattern from data.py:138-149
        mp4_bytes = io.BytesIO(data["video"])
        vr = decord.VideoReader(mp4_bytes, ctx=decord.cpu(0), num_threads=1)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

        # Serialize actions to msgpack bytes to avoid grain batching issues
        # with complex nested dicts (actions is a list of dicts with mouse, keyboard, etc.)
        actions = data.get("actions")
        actions_bytes = msgpack.packb(actions, use_bin_type=True) if actions is not None else None

        return {
            "videos": frames.astype(np.float32),  # (T, H, W, C) in [0, 255]
            "actions_bytes": actions_bytes,  # Serialized to avoid batching issues
            "source": data.get("source"),
        }


def make_tokenization_iterator(
    input_dir: str,
    num_shards: int | None,
    batch_size: int,
    num_workers: int,
):
    """Create dataloader for tokenization (sequential, no shuffle, full episodes)."""
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
        MinecraftVPTProcessFullEpisode(),
        grain.transforms.Batch(batch_size=batch_size, drop_remainder=False),
    ]

    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=1,
    )


# ==============================================================================
# Shard Writing
# ==============================================================================


class ShardWriter:
    """Write records to output shards, maintaining records_per_shard records per shard."""

    def __init__(self, output_dir: Path, records_per_shard: int = 1000):
        self.output_dir = output_dir
        self.records_per_shard = records_per_shard
        self.writer = None
        self.shard_idx = 0
        self.records_in_shard = 0
        self.total_records = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _open_new_shard(self):
        if self.writer is not None:
            self.writer.close()
        path = self.output_dir / f"shard-{self.shard_idx:05d}.array_record"
        self.writer = ArrayRecordWriter(str(path), "group_size:1")
        self.shard_idx += 1
        self.records_in_shard = 0

    def write(self, record: dict):
        if self.writer is None or self.records_in_shard >= self.records_per_shard:
            self._open_new_shard()
        self.writer.write(serialize_record(record))
        self.records_in_shard += 1
        self.total_records += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ==============================================================================
# Main
# ==============================================================================


def parse_args():
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
        default="/home/ubuntu/minecraft-vpt/arrayrecords-mp4",
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
        default=8,
        help="Videos to encode per batch",
    )
    parser.add_argument(
        "--packing_factor",
        type=int,
        default=None,
        help="Optional latent packing factor",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Grain dataloader workers",
    )
    parser.add_argument(
        "--records_per_shard",
        type=int,
        default=1000,
        help="Number of records per output shard",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[tokenize] Loading tokenizer from: {args.tokenizer_ckpt}")
    print(f"[tokenize] Input directory: {args.input_dir}")
    print(f"[tokenize] Output directory: {args.output_dir}")

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

        # Create dataloader
        dataloader = make_tokenization_iterator(
            input_dir=args.input_dir,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Create shard writer
        output_dir = Path(args.output_dir)
        writer = ShardWriter(output_dir, records_per_shard=args.records_per_shard)

        # JIT-compile encode function
        @nnx.jit
        def encode_batch(videos):
            latents, _ = tokenizer.encode(
                videos,
                packing_factor=args.packing_factor,
                deterministic=True,
                rngs=nnx.Rngs(0),
            )
            return latents

        # Process all batches
        total_videos = 0
        try:
            for batch in tqdm(dataloader, desc="Tokenizing"):
                videos = batch["videos"]  # (B, T, H, W, C)
                actions_bytes_batch = batch["actions_bytes"]  # (B,) - each is msgpack bytes
                sources_batch = batch["source"]  # (B,)

                batch_size = videos.shape[0]

                # Pad batch to be divisible by device count for proper sharding
                num_devices = jax.device_count()
                pad_size = (num_devices - batch_size % num_devices) % num_devices
                if pad_size > 0:
                    # Pad with zeros (or repeat last video)
                    padding = np.zeros((pad_size,) + videos.shape[1:], dtype=videos.dtype)
                    videos = np.concatenate([videos, padding], axis=0)

                videos_device = jax.device_put(videos, data_sharding)
                latents = encode_batch(videos_device)
                latents_np = np.asarray(latents)  # (B + pad, T, n_latents, d_bottleneck)

                # Remove padding
                if pad_size > 0:
                    latents_np = latents_np[:batch_size]

                # Write each record individually
                for i in range(batch_size):
                    record = {
                        "latents": latents_np[i],  # (T, n_latents, d_bottleneck)
                        "actions_packed": actions_bytes_batch[i],  # Pre-packed msgpack bytes
                        "source": sources_batch[i] if sources_batch is not None else None,
                    }
                    writer.write(record)

                total_videos += batch_size

        finally:
            writer.close()

        print(f"[tokenize] Done! Processed {total_videos} videos")
        print(f"[tokenize] Wrote {writer.shard_idx} shards to {args.output_dir}")
        print(f"[tokenize] Total records: {writer.total_records}")


if __name__ == "__main__":
    main()
