"""
Streaming HuggingFace data source for Minecraft VPT dataset.

Implements Grain's RandomAccessDataSource protocol with on-demand shard downloading.
"""

import bisect
import json
import shutil
import tarfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import grain
from huggingface_hub import hf_hub_download


@dataclass
class MinecraftVPTConfig:
    """Configuration for MinecraftVPT data source."""

    repo_id: str = "p-doom/openai-minecraft-dataset"
    target_dir: str = "datasets/minecraft_vpt"
    max_shards: int = 4


class MinecraftVPTDataSource:
    """
    RandomAccessDataSource that downloads shards on-demand from HuggingFace.

    Thread-safe and pickleable for Grain multi-worker prefetch.

    Usage:
        config = MinecraftVPTConfig(max_shards=4)
        source = MinecraftVPTDataSource(config)
        print(len(source))  # Total records across all shards
        record = source[0]  # Triggers download of first shard if needed
    """

    def __init__(self, config: MinecraftVPTConfig):
        self.config = config
        self.target_dir = Path(config.target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe shard readers (lazy-loaded)
        self._lock = threading.Lock()
        self._shard_sources: dict[int, grain.sources.ArrayRecordDataSource] = {}

        # Build index (downloads shards if needed to count records)
        self._build_index()

    def _build_index(self):
        """Build global_idx -> (shard_id, local_idx) mapping.

        On first run, downloads and extracts all configured shards to count records.
        Caches the counts to a JSON file for subsequent runs.
        """
        metadata_file = self.target_dir / "shard_metadata.json"

        # Try to load cached metadata
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            # Verify all needed shards are present in metadata
            if all(str(i) in metadata for i in range(self.config.max_shards)):
                self._shard_offsets = []
                self._total_records = 0
                for i in range(self.config.max_shards):
                    self._shard_offsets.append(self._total_records)
                    self._total_records += metadata[str(i)]
                return

        # Download and count each shard
        metadata = {}
        self._shard_offsets = []
        self._total_records = 0
        for shard_id in range(self.config.max_shards):
            print(f"Indexing shard {shard_id + 1}/{self.config.max_shards}...")
            shard_dir = self._ensure_shard_downloaded(shard_id)
            paths = sorted(str(p) for p in shard_dir.glob("*.array_record"))
            source = grain.sources.ArrayRecordDataSource(paths)
            count = len(source)
            metadata[str(shard_id)] = count
            self._shard_offsets.append(self._total_records)
            self._total_records += count
            self._shard_sources[shard_id] = source
            print(f"  Shard {shard_id}: {count} records")

        # Cache metadata for future runs
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        print(f"Total records: {self._total_records}")

    def _ensure_shard_downloaded(self, shard_id: int) -> Path:
        """Download and extract shard if not present."""
        shard_dir = self.target_dir / f"shard_{shard_id:05d}"

        # Check if already extracted
        if shard_dir.exists() and list(shard_dir.glob("*.array_record")):
            return shard_dir

        # Download tar from HuggingFace
        tar_name = f"shard_{shard_id:05d}.tar"
        print(f"Downloading {tar_name} from {self.config.repo_id}...")
        tar_path = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=tar_name,
            repo_type="dataset",
        )

        # Extract to temp dir, then atomic rename
        temp_dir = self.target_dir / f".tmp_{shard_id:05d}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {tar_name}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(temp_dir)

        # Move extracted contents to shard_dir
        # The tar may contain a subdirectory, so we handle that
        extracted_items = list(temp_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            # Tar contained a single directory - move its contents
            inner_dir = extracted_items[0]
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            inner_dir.rename(shard_dir)
            temp_dir.rmdir()
        else:
            # Tar extracted directly - rename temp_dir
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            temp_dir.rename(shard_dir)

        return shard_dir

    def _get_shard_source(self, shard_id: int) -> grain.sources.ArrayRecordDataSource:
        """Get or create ArrayRecordDataSource for shard (thread-safe)."""
        with self._lock:
            if shard_id not in self._shard_sources:
                shard_dir = self._ensure_shard_downloaded(shard_id)
                paths = sorted(str(p) for p in shard_dir.glob("*.array_record"))
                self._shard_sources[shard_id] = grain.sources.ArrayRecordDataSource(
                    paths
                )
            return self._shard_sources[shard_id]

    def __len__(self) -> int:
        return self._total_records

    def __getitem__(self, idx: int) -> bytes:
        if idx < 0 or idx >= self._total_records:
            raise IndexError(f"Index {idx} out of range [0, {self._total_records})")

        # Binary search to find shard
        shard_id = bisect.bisect_right(self._shard_offsets, idx) - 1
        local_idx = idx - self._shard_offsets[shard_id]

        source = self._get_shard_source(shard_id)
        return source[local_idx]

    def __repr__(self) -> str:
        return (
            f"MinecraftVPTDataSource("
            f"shards={self.config.max_shards}, "
            f"total={self._total_records})"
        )

    def __getstate__(self):
        """Pickle: exclude non-pickleable objects (lock, open file handles)."""
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_shard_sources"] = {}
        return state

    def __setstate__(self, state):
        """Unpickle: recreate lock, sources will be lazy-loaded."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
