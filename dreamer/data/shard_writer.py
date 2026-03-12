"""ArrayRecord shard writer with automatic rotation.

Writes msgpack-serialized records to ArrayRecord shards,
rotating to a new shard file after records_per_shard records.
"""

from array_record.python.array_record_module import ArrayRecordWriter
from pathlib import Path

from .serialization import serialize_msgpack_record


class ShardWriter:
    """Write records to output shards with automatic rotation.

    Example:
        >>> with ShardWriter(output_dir, records_per_shard=1000) as writer:
        ...     for record in records:
        ...         writer.write(record)
        >>> print(f"Wrote {writer.total_records} records to {writer.num_shards} shards")
    """

    def __init__(
        self,
        output_dir: Path | str,
        records_per_shard: int = 1000,
        start_shard_idx: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.records_per_shard = records_per_shard

        self.writer = None
        self.shard_idx = start_shard_idx
        self.records_in_shard = 0
        self._total_records = 0
        self._completed_shards: list[str] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _open_new_shard(self) -> None:
        """Open a new shard file, closing the previous one if it exists."""
        if self.writer is not None:
            self.writer.close()
            self._completed_shards.append(
                str(self.output_dir / f"shard-{self.shard_idx - 1:05d}.array_record")
            )
        path = self.output_dir / f"shard-{self.shard_idx:05d}.array_record"
        self.writer = ArrayRecordWriter(str(path), "group_size:1")
        self.shard_idx += 1
        self.records_in_shard = 0

    def write(self, record: dict) -> None:
        if self.writer is None or self.records_in_shard >= self.records_per_shard:
            self._open_new_shard()

        serialized = serialize_msgpack_record(record)
        self.writer.write(serialized)
        self.records_in_shard += 1
        self._total_records += 1

    def drain_completed(self) -> list[str]:
        """Return and clear the list of fully closed shard file paths.

        Safe to call only from the thread that calls write() / close().
        """
        completed = self._completed_shards[:]
        self._completed_shards.clear()
        return completed

    def close(self) -> None:
        """Close the current shard writer."""
        if self.writer is not None:
            self.writer.close()
            self._completed_shards.append(
                str(self.output_dir / f"shard-{self.shard_idx - 1:05d}.array_record")
            )
            self.writer = None

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
