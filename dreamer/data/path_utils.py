"""Path utilities for ArrayRecord dataset discovery and generation."""

import os
from typing import Literal


def discover_array_record_paths(path: str | list[str]) -> list[str]:
    """Discover .array_record files from directory or file list.

    Args:
        path: Either a directory path, a single file path, or a list of file paths

    Returns:
        List of absolute paths to .array_record files
    """
    if isinstance(path, list):
        return path

    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".array_record")
        ]
    else:
        return [path]


def discover_shard_paths(path: str | list[str], prefix: str = "shard") -> list[str]:
    """Discover shard files named like shard-XXXXX.array_record."""
    if isinstance(path, list):
        return sorted(path)

    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.startswith(f"{prefix}-") and f.endswith(".array_record")
        )

    return [path]


def build_dataset_paths(
    array_record_path: str | list[str],
    dataset_type: Literal["coinrun", "minecraft_vpt", "latent"],
    index_max: int | None = None,
) -> list[str]:
    """Unified path builder for all dataset types.

    Args:
        array_record_path: Path or list of paths to ArrayRecord files/directories
        dataset_type: Type of dataset ("coinrun", "minecraft_vpt", "latent")
        index_max: For minecraft_vpt/latent, max number of shards to load; None/<=0 means all

    Returns:
        List of paths to ArrayRecord files

    """
    # Minecraft VPT and latent datasets are shard-based.
    if dataset_type in ("minecraft_vpt", "latent"):
        shard_paths = discover_shard_paths(array_record_path)
        if not shard_paths:
            raise ValueError(f"No shards found for {dataset_type} at {array_record_path}")
        shard_paths = sorted(shard_paths)
        if index_max is None or index_max <= 0:
            return shard_paths
        return shard_paths[: min(index_max, len(shard_paths))]

    # CoinRun uses file discovery
    return discover_array_record_paths(array_record_path)
