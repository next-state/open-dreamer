"""Unified ArrayRecord shard writer with automatic rotation.

Provides a consistent interface for writing records to ArrayRecord shards
with automatic rotation based on records_per_shard limit.
"""

from array_record.python.array_record_module import ArrayRecordWriter
from pathlib import Path
from typing import Literal

from .serialization import serialize_pickle_record, serialize_msgpack_record


class ShardWriter:
    """Write records to output shards with automatic rotation.

    Maintains a consistent number of records per shard, automatically
    creating new shards when the limit is reached.

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
        serialization_format: Literal["pickle", "msgpack"] = "msgpack",
    ):
        """Initialize shard writer.

        Args:
            output_dir: Directory to write shards to
            records_per_shard: Maximum records per shard before rotation
            serialization_format: Serialization format ("pickle" or "msgpack")
        """
        self.output_dir = Path(output_dir)
        self.records_per_shard = records_per_shard
        self.serialization_format = serialization_format

        self.writer = None
        self.shard_idx = 0
        self.records_in_shard = 0
        self._total_records = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Select serializer
        if serialization_format == "pickle":
            self._serializer = serialize_pickle_record
        elif serialization_format == "msgpack":
            self._serializer = serialize_msgpack_record
        else:
            raise ValueError(f"Unknown serialization format: {serialization_format}")

    def _open_new_shard(self) -> None:
        """Open a new shard file, closing the previous one if it exists."""
        if self.writer is not None:
            self.writer.close()
        path = self.output_dir / f"shard-{self.shard_idx:05d}.array_record"
        self.writer = ArrayRecordWriter(str(path), "group_size:1")
        self.shard_idx += 1
        self.records_in_shard = 0

    def write(self, record: dict) -> None:
        """Write a single record, rotating to new shard if needed.

        Args:
            record: Dictionary to serialize and write
        """
        if self.writer is None or self.records_in_shard >= self.records_per_shard:
            self._open_new_shard()

        serialized = self._serializer(record)
        self.writer.write(serialized)
        self.records_in_shard += 1
        self._total_records += 1

    def close(self) -> None:
        """Close the current shard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "ShardWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures writer is closed."""
        self.close()

    @property
    def total_records(self) -> int:
        """Total number of records written across all shards."""
        return self._total_records

    @property
    def num_shards(self) -> int:
        """Number of shards created (including current open shard)."""
        return self.shard_idx
