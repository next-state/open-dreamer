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
from pathlib import Path

import decord
import grain
import jax
import numpy as np
from flax import nnx
from tqdm import tqdm

from dreamer.actions import parse_action_dicts
from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.parallel import build_parallel
from dreamer.data.transforms import ProcessMinecraftEpisodeAndSlice
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


# ==============================================================================
# Data Loading
# ==============================================================================


class MinecraftVPTProcessFullEpisode(grain.transforms.Map):
    """Decode full MP4 video for tokenization (no slicing).

    Uses ProcessMinecraftEpisodeAndSlice with full_episode=True, then adds action parsing.
    """

    def __init__(self):
        # Use the shared transform with full_episode mode
        # Note: Map needs random_map to be called with rng, but we'll handle that
        self.processor = ProcessMinecraftEpisodeAndSlice(
            seq_len=0,  # Ignored when full_episode=True
            image_h=128,  # Dummy values
            image_w=128,
            image_c=3,
            full_episode=True,
        )

    def map(self, element: bytes) -> dict:
        import pickle
        # Create a dummy RNG for the processor
        rng = np.random.default_rng(0)

        # Get video from shared processor
        result = self.processor.random_map(element, rng)

        # Parse actions from original data
        data = pickle.loads(element)
        actions = data.get("actions")
        actions = parse_action_dicts(actions).to_dict()

        return {
            "videos": result["videos"],  # (T, H, W, C) in [0, 255]
            "actions": actions,
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


class DeviceShardedIterator:
    """Wraps a dataloader to yield device-sharded JAX arrays.

    Handles padding and direct per-device transfers to avoid GPU 0 memory spikes.
    """

    def __init__(
        self,
        dataloader,
        sharding: jax.sharding.NamedSharding,
        sharded_keys: tuple[str, ...] = ("videos",),
    ):
        self.dataloader = dataloader
        self.sharding = sharding
        self.sharded_keys = sharded_keys
        self.devices = list(sharding.mesh.devices.flat)
        self.num_devices = len(self.devices)

    def _shard_array(self, array: np.ndarray) -> tuple[jax.Array, int]:
        """Shard array across devices, returning (sharded_array, pad_size)."""
        batch_size = array.shape[0]
        pad_size = (self.num_devices - batch_size % self.num_devices) % self.num_devices

        if pad_size > 0:
            padding = np.zeros((pad_size,) + array.shape[1:], dtype=array.dtype)
            array = np.concatenate([array, padding], axis=0)

        # Transfer each shard directly to its target device
        per_device = array.shape[0] // self.num_devices
        shards = [
            jax.device_put(array[i * per_device : (i + 1) * per_device], d)
            for i, d in enumerate(self.devices)
        ]
        sharded = jax.make_array_from_single_device_arrays(
            array.shape, self.sharding, shards
        )
        return sharded, pad_size

    def __iter__(self):
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


# ==============================================================================
# Shard Writing (imported from shared module)
# ==============================================================================

# ShardWriter is imported at the top of the file


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

        # Create dataloader with device sharding
        base_dataloader = make_tokenization_iterator(
            input_dir=args.input_dir,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dataloader = DeviceShardedIterator(
            base_dataloader,
            sharding=data_sharding,
            sharded_keys=("videos",),
        )

        # Create shard writer (using msgpack serialization)
        output_dir = Path(args.output_dir)
        writer = ShardWriter(
            output_dir,
            records_per_shard=args.records_per_shard,
            serialization_format="msgpack"
        )

        # JIT-compile encode function
        @nnx.jit
        def encode_batch(videos):
            # Returns unpacked latents: (B, T, n_latents, d_bottleneck)
            latents, _ = tokenizer.encode(
                videos,
                deterministic=True,
                rngs=nnx.Rngs(0),
            )
            return latents

        # Process all batches
        total_videos = 0
        all_channel_means = []
        all_channel_stds = []

        try:
            pbar = tqdm(dataloader, desc="Tokenizing")
            for batch in pbar:
                videos = batch["videos"]  # Already sharded JAX array
                actions_batch = batch["actions"]
                sources_batch = batch["source"]
                batch_size = batch["_batch_size"]
                pad_size = batch["_pad_size"]

                latents = encode_batch(videos)

                # Compute per-channel stats on GPU (over B, T, n_latents)
                channel_mean = latents[:batch_size].mean(axis=(0, 1, 2))
                channel_std = latents[:batch_size].std(axis=(0, 1, 2))
                all_channel_means.append(np.asarray(channel_mean))
                all_channel_stds.append(np.asarray(channel_std))

                # Update tqdm with scalar stats
                pbar.set_postfix(mean=f"{float(channel_mean.mean()):.4f}", std=f"{float(channel_std.mean()):.4f}")

                # Gather latents from each device shard separately to avoid GPU 0 spike
                latents_np = np.concatenate(
                    [np.asarray(shard.data) for shard in latents.addressable_shards],
                    axis=0,
                )

                # Remove padding
                if pad_size > 0:
                    latents_np = latents_np[:batch_size]

                # Write each record individually
                for i in range(batch_size):
                    # Extract i-th element from each array in the actions dict (handle None)
                    actions_i = {k: v[i] if v is not None else None for k, v in actions_batch.items()}
                    record = {
                        "latents": latents_np[i],  # (T, n_latents, d_bottleneck)
                        "actions": actions_i,
                        "source": sources_batch[i] if sources_batch is not None else None,
                    }
                    writer.write(record)

                total_videos += batch_size

        finally:
            writer.close()

        print(f"[tokenize] Done! Processed {total_videos} videos")
        print(f"[tokenize] Wrote {writer.shard_idx} shards to {args.output_dir}")
        print(f"[tokenize] Total records: {writer.total_records}")
        
        # Print final channel-wise statistics
        if all_channel_means:
            mean_of_means = np.stack(all_channel_means).mean(axis=0)
            mean_of_stds = np.stack(all_channel_stds).mean(axis=0)
            
            print(f"\n[tokenize] Latent statistics ({len(all_channel_means)} batches):")
            print(f"[tokenize] Channel-wise mean: {mean_of_means}")
            print(f"[tokenize] Channel-wise std: {mean_of_stds}")


if __name__ == "__main__":
    main()
