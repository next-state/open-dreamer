"""
Tokenize Minecraft VPT dataset.

Reads existing ArrayRecord shards containing MP4-encoded Minecraft VPT videos,
decodes them, encodes them into latents using a pretrained tokenizer, and writes
new ArrayRecord shards with latents (using msgpack serialization).

Pipeline per input shard:
  [GCS Download] → [Grain DataLoader] → [CPU Prefetch] → [Device Transfer + Prefetch]
                 → [Main Loop] → [Async Writer] → [GCS Upload]

Input shards are downloaded one at a time to /tmp (with 1 shard pre-fetched ahead),
processed, then deleted. Output shards are written locally to a staging dir and
uploaded to GCS as soon as each output shard is finalized.

Example usage:
    python scripts/tokenize_minecraft_dataset.py \
        tokenizer_ckpt=/path/to/tokenizer/ckpt \
        dataset.array_record_path=gs://bucket/arrayrecords-mp4 \
        output_dir=gs://bucket/latents-out
"""
import logging
import os
import pickle
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

import decord
import grain
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig
from tqdm import tqdm

from dreamer.checkpointing import TokenizerCheckpointBundle
from dreamer.data.shard_writer import ShardWriter
from dreamer.data.transforms import ProcessMinecraftEpisodeAndSlice
from dreamer.parallel import build_parallel

decord.bridge.set_bridge("native")

logging.getLogger("absl").setLevel(logging.WARNING)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'


# ==============================================================================
# Shard listing
# ==============================================================================


def list_shards(input_dir: str, index_max: int | None, index_min: int = 0) -> list[str]:
    """List shard paths from a local directory or GCS prefix."""
    if input_dir.startswith("gs://"):
        result = subprocess.run(
            ["gsutil", "ls", f"{input_dir.rstrip('/')}/shard-*.array_record"],
            capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
        )
        all_shards = sorted(result.stdout.strip().splitlines())
    else:
        all_shards = sorted(str(p) for p in Path(input_dir).glob("shard-*.array_record"))
    if index_max is not None:
        all_shards = all_shards[:index_max]
    if index_min:
        all_shards = all_shards[index_min:]
    if not all_shards:
        raise ValueError(f"No shards found in {input_dir}")
    return all_shards


# ==============================================================================
# Download-ahead pipeline
# ==============================================================================


class DownloadAheadPipeline:
    """Downloads GCS shards to local tmp ahead of processing.

    Keeps `lookahead` shards pre-downloaded on disk. Deletes each shard
    after the caller has finished with it (on iterator advance).
    Works transparently for local paths too (yields path directly, no copy).
    """

    def __init__(self, shard_paths: list[str], tmp_dir: str, lookahead: int = 1):
        self._paths = shard_paths
        self._tmp_dir = Path(tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._is_gcs = any(p.startswith("gs://") for p in shard_paths[:1])
        # Queue holds (local_path, is_tmp) — is_tmp=True means we should delete after use
        self._ready: queue.Queue = queue.Queue(maxsize=lookahead)
        self._thread = Thread(target=self._download_loop, daemon=True)
        self._thread.start()

    def _download_one(self, gcs_path: str) -> str:
        local = str(self._tmp_dir / Path(gcs_path).name)
        for attempt in range(3):
            try:
                subprocess.run(
                    ["gsutil", "-q", "cp", gcs_path, local],
                    check=True, capture_output=True, stdin=subprocess.DEVNULL,
                )
                return local
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def _download_loop(self):
        try:
            for path in self._paths:
                if self._is_gcs:
                    local = self._download_one(path)
                    self._ready.put((local, True))
                else:
                    self._ready.put((path, False))
        except Exception as e:
            self._ready.put((e, False))
        finally:
            self._ready.put((None, False))  # sentinel

    def __iter__(self):
        while True:
            local_path, is_tmp = self._ready.get()
            if local_path is None:
                break
            if isinstance(local_path, Exception):
                raise local_path
            try:
                yield local_path
            finally:
                if is_tmp:
                    Path(local_path).unlink(missing_ok=True)


# ==============================================================================
# Async GCS uploader
# ==============================================================================


class AsyncGCSUploader:
    """Uploads local shard files to GCS in a background thread, deletes after success."""

    def __init__(self, gcs_dir: str):
        self._gcs_dir = gcs_dir.rstrip("/")
        self._queue: queue.Queue = queue.Queue()
        self._errors: list[str] = []
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _upload_one(self, local_path: str):
        gcs_path = f"{self._gcs_dir}/{Path(local_path).name}"
        for attempt in range(3):
            try:
                subprocess.run(
                    ["gsutil", "-q", "-o", "GSUtil:parallel_composite_upload_threshold=100M",
                     "cp", local_path, gcs_path],
                    check=True, capture_output=True, stdin=subprocess.DEVNULL,
                )
                size_mb = Path(local_path).stat().st_size / 1024**2
                Path(local_path).unlink(missing_ok=True)
                print(f"[tokenize] Uploaded {Path(local_path).name} ({size_mb:.0f} MB) → {self._gcs_dir}")
                return
            except subprocess.CalledProcessError as e:
                if attempt == 2:
                    self._errors.append(f"Failed to upload {local_path}: {e.stderr.decode()[:200]}")
                    return
                time.sleep(2 ** attempt)

    def _loop(self):
        for item in iter(self._queue.get, None):
            if isinstance(item, threading.Event):
                item.set()
            else:
                self._upload_one(item)

    def upload(self, local_path: str):
        self._queue.put(local_path)

    def flush(self):
        """Block until all currently queued uploads are complete."""
        event = threading.Event()
        self._queue.put(event)
        event.wait()
        if self._errors:
            raise RuntimeError(f"Upload errors: {self._errors}")

    def close(self):
        self.flush()
        self._queue.put(None)
        self._thread.join()
        if self._errors:
            raise RuntimeError(f"Upload errors: {self._errors}")


# ==============================================================================
# Async Shard Writer
# ==============================================================================


class AsyncShardWriter:
    """Wraps ShardWriter with a background thread so disk I/O never blocks the main loop."""

    def __init__(
        self,
        output_dir: Path | str,
        records_per_shard: int = 5000,
        maxsize: int = 200,
        start_shard_idx: int = 0,
    ):
        self._writer = ShardWriter(output_dir, records_per_shard, start_shard_idx=start_shard_idx)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self):
        for item in iter(self._queue.get, None):
            if isinstance(item, threading.Event):
                item.set()
            else:
                self._writer.write(item)

    def write(self, record: dict):
        self._queue.put(record)

    def sync(self):
        """Block until all queued writes have been processed by the background thread."""
        event = threading.Event()
        self._queue.put(event)
        event.wait()

    def drain_completed(self) -> list[str]:
        """Return completed (fully closed) shard file paths. Call after sync()."""
        return self._writer.drain_completed()

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
# Pre-decode pipeline (thread-pool, no /dev/shm)
# ==============================================================================


def predecode_shard(
    local_path: str,
    transform: ProcessMinecraftEpisodeAndSlice,
    num_threads: int,
) -> list[dict]:
    """Pre-decode all episodes in a local ArrayRecord shard using a thread pool.

    Uses threads (not processes) so there is no /dev/shm constraint. decord releases
    the GIL during MP4 decoding, enabling true parallelism across threads.
    Returns a list of decoded episode dicts, all in regular RAM.
    """
    source = grain.sources.ArrayRecordDataSource([local_path])
    n = len(source)

    # Read raw bytes sequentially (fast, ArrayRecord is optimised for sequential I/O)
    raw_records = [source[i] for i in range(n)]

    dummy_rng = np.random.default_rng(0)

    def decode_one(raw: bytes) -> dict:
        return transform.random_map(raw, dummy_rng)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        episodes = list(ex.map(decode_one, raw_records))
    dt = time.perf_counter() - t0
    print(f"[tokenize] Pre-decoded {n} episodes in {dt:.1f}s  ({dt/n:.2f}s/ep, {num_threads} threads)")
    return episodes


def make_device_prefetcher(
    episodes: list[dict],
    batch_size: int,
    sharding: jax.sharding.NamedSharding,
    prefetch: int = 2,
):
    """Batch episodes, transfer to device in a background thread, yield ready batches.

    Overlaps H2D transfer of batch N+1 with GPU encoding of batch N.
    Episodes with drop_remainder (same as grain Batch default).
    """
    devices = list(sharding.mesh.devices.flat)
    num_devices = len(devices)

    # Build batches (drop remainder)
    n_full = (len(episodes) // batch_size) * batch_size
    batches_raw = [episodes[i:i + batch_size] for i in range(0, n_full, batch_size)]

    def stack_and_transfer(group: list[dict]) -> dict:
        videos = np.stack([ep["videos"] for ep in group])  # (B, T, H, W, C)
        B = videos.shape[0]
        pad_size = (num_devices - B % num_devices) % num_devices
        if pad_size > 0:
            videos = np.concatenate(
                [videos, np.zeros((pad_size,) + videos.shape[1:], dtype=videos.dtype)], axis=0
            )
        per_device = videos.shape[0] // num_devices
        shards = [
            jax.device_put(videos[i * per_device:(i + 1) * per_device], devices[i]).astype(jnp.bfloat16)
            for i in range(num_devices)
        ]
        sharded = jax.make_array_from_single_device_arrays(videos.shape, sharding, shards)
        return {
            "videos": sharded,
            "actions": {k: np.stack([ep["actions"][k] for ep in group])
                        if group[0]["actions"].get(k) is not None else None
                        for k in group[0]["actions"]},
            "source": [ep.get("source") for ep in group],
            "_batch_size": B,
            "_pad_size": pad_size,
        }

    ready: queue.Queue = queue.Queue(maxsize=prefetch)

    def _transfer_loop():
        for group in batches_raw:
            ready.put(stack_and_transfer(group))
        ready.put(None)

    t = Thread(target=_transfer_loop, daemon=True)
    t.start()
    return ready


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
    output_dir_str = str(cfg.output_dir).rstrip("/")
    is_gcs_output = output_dir_str.startswith("gs://")

    # Always write locally to a staging dir; upload to GCS from there
    staging_dir = Path("/tmp/tokenize_output") if is_gcs_output else Path(output_dir_str)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "metadata").mkdir(exist_ok=True)
    stats_path = str(staging_dir / "metadata" / "latent_stats.npz")

    print(f"[tokenize] Loading tokenizer from: {cfg.tokenizer_ckpt}")
    print(f"[tokenize] Input: {cfg.dataset.array_record_path}")
    print(f"[tokenize] Output: {output_dir_str}  (staging: {staging_dir})")

    dataloader_cfg = cfg.dataset.dataloader_cfg
    mesh, data_sharding, mesh_rules = build_parallel(cfg.parallel_strategy)

    # List all input shards upfront
    shard_paths = list_shards(
        cfg.dataset.array_record_path,
        cfg.dataset.index_max,
        index_min=int(getattr(cfg.dataset, "index_min", 0)),
    )
    print(f"[tokenize] Found {len(shard_paths)} input shards")

    with jax.set_mesh(mesh):
        bundle = TokenizerCheckpointBundle.from_pretrained(
            cfg.tokenizer_ckpt,
            mesh_rules=mesh_rules,
        )
        tokenizer = bundle.tokenizer
        del tokenizer.decoder

        print(f"[tokenize] Tokenizer loaded. "
              f"n_latents={tokenizer.encoder.n_latents}, "
              f"d_bottleneck={tokenizer.cfg.encoder.d_bottleneck}, "
              f"patch_size={tokenizer.cfg.encoder.patch_size}")

        start_shard_idx = int(getattr(cfg, "start_shard_idx", 0))
        writer = AsyncShardWriter(staging_dir, records_per_shard=cfg.records_per_shard, start_shard_idx=start_shard_idx)
        uploader = AsyncGCSUploader(output_dir_str) if is_gcs_output else None

        @nnx.jit
        def encode_batch(videos):
            latents, _ = tokenizer.encode(videos, deterministic=True, rngs=nnx.Rngs(0))
            return latents

        n_channels = tokenizer.cfg.encoder.d_bottleneck
        welford = WelfordAccumulator(n_channels)
        total_videos = 0
        total_batches = 0

        # How many threads to use for parallel MP4 decoding.
        # Each thread decodes one episode independently; decord releases the GIL.
        host_cpus = os.cpu_count() or 8
        predecode_threads = max(1, min(host_cpus - 4, 64))
        decoder_threads_per_worker = max(1, min(4, host_cpus // max(1, predecode_threads)))
        print(f"[tokenize] Pre-decode: {predecode_threads} threads, "
              f"{decoder_threads_per_worker} decord threads/ep")

        transform = ProcessMinecraftEpisodeAndSlice(
            seq_len=0,
            image_h=cfg.dataset.H,
            image_w=cfg.dataset.W,
            image_c=cfg.dataset.C,
            padding_h=cfg.dataset.padding_H,
            padding_w=cfg.dataset.padding_W,
            patch_size=cfg.dataset.patch_size,
            full_episode=True,
            decoder_threads=decoder_threads_per_worker,
            cast_to_float32=bool(getattr(dataloader_cfg, "cast_to_float32", False)),
        )

        def _drain_and_upload():
            if uploader is not None:
                for p in writer.drain_completed():
                    uploader.upload(p)

        def _save_stats():
            np.savez(
                stats_path,
                mean=welford.mean.astype(np.float32),
                std=welford.std.astype(np.float32),
                num_samples=welford.count,
                num_videos=total_videos,
            )
            if is_gcs_output:
                subprocess.run(
                    ["gsutil", "-q", "cp", stats_path,
                     f"{output_dir_str}/metadata/latent_stats.npz"],
                    check=False, stdin=subprocess.DEVNULL,
                )

        try:
            downloader = DownloadAheadPipeline(
                shard_paths,
                tmp_dir="/tmp/tokenize_shards",
                lookahead=2,  # 2 shards pre-downloaded (~1.3 GB on disk)
            )
            for shard_num, local_shard_path in enumerate(downloader):
                shard_name = Path(local_shard_path).name
                print(f"\n[tokenize] Shard {shard_num + 1}/{len(shard_paths)}: {shard_name}")

                # Pre-decode all episodes in the shard in parallel (threads, no /dev/shm)
                episodes = predecode_shard(local_shard_path, transform, predecode_threads)

                # Batch + H2D transfer in background thread, overlapping with GPU encode
                ready_queue = make_device_prefetcher(
                    episodes,
                    batch_size=dataloader_cfg.B,
                    sharding=data_sharding,
                    prefetch=2,
                )

                pbar = tqdm(total=len(episodes) // dataloader_cfg.B, desc="  Encoding")
                t_iter_start = time.perf_counter()

                while True:
                    batch = ready_queue.get()
                    if batch is None:
                        break
                    t_iter = time.perf_counter() - t_iter_start

                    videos = batch["videos"]
                    actions_batch = batch["actions"]
                    sources_batch = batch["source"]
                    batch_size = batch["_batch_size"]

                    t0 = time.perf_counter()
                    latents = encode_batch(videos)
                    latents.block_until_ready()
                    t1 = time.perf_counter()

                    latents_np = np.asarray(latents)[:batch_size]
                    t2 = time.perf_counter()

                    welford.update(latents_np.astype(np.float32).reshape(-1, n_channels))
                    t3 = time.perf_counter()

                    for i in range(batch_size):
                        actions_i = {k: v[i] if v is not None else None for k, v in actions_batch.items()}
                        writer.write({
                            "latents": latents_np[i],
                            "actions": actions_i,
                            "source": sources_batch[i] if sources_batch is not None else None,
                        })
                    t4 = time.perf_counter()

                    pbar.set_postfix(
                        mean=f"{welford.mean.mean():.4f}",
                        std=f"{welford.std.mean():.4f}",
                        fetch=f"{t_iter:.3f}s",
                        encode=f"{t1 - t0:.3f}s",
                        d2h=f"{t2 - t1:.3f}s",
                        queue=f"{t4 - t3:.3f}s",
                    )

                    total_videos += batch_size
                    total_batches += 1
                    pbar.update(1)
                    t_iter_start = time.perf_counter()

                    # Save stats every 100 batches
                    if total_batches % 100 == 0:
                        _save_stats()

                pbar.close()
                del episodes  # free ~10-70 GB of decoded frames

                # After each input shard: flush writer, upload any completed output shards
                writer.sync()
                _drain_and_upload()

                # Log local disk usage
                staging_gb = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file()) / 1024**3
                total_gb = shutil.disk_usage("/").used / 1024**3
                free_gb = shutil.disk_usage("/").free / 1024**3
                print(f"[tokenize] Disk: staging={staging_gb:.1f} GB used, "
                      f"total_used={total_gb:.1f} GB, free={free_gb:.1f} GB")

        finally:
            writer.close()
            _drain_and_upload()
            if uploader is not None:
                uploader.flush()  # wait for all uploads to finish

            if welford.count > 0:
                _save_stats()
                # Ensure final stats are uploaded
                if is_gcs_output:
                    subprocess.run(
                        ["gsutil", "-q", "cp", stats_path,
                         f"{output_dir_str}/metadata/latent_stats.npz"],
                        check=True, stdin=subprocess.DEVNULL,
                    )

            if uploader is not None:
                uploader.close()

        print(f"[tokenize] Done! Processed {total_videos} videos")
        print(f"[tokenize] Wrote {writer.shard_idx} output shards to {output_dir_str}")
        print(f"[tokenize] Total records: {writer.total_records}")
        print(f"[tokenize] Latent stats: mean={welford.mean.mean():.4f}, std={welford.std.mean():.4f}")


@hydra.main(version_base=None, config_path="../configs", config_name="tokenize")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
