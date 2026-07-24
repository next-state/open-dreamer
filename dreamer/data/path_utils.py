"""Path utilities for ArrayRecord dataset discovery and generation."""

import os
from pathlib import Path
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


def generate_shard_paths(
    base_dir: str,
    num_shards: int,
    prefix: str = "shard"
) -> list[str]:
    """Generate shard paths with consistent naming.

    Args:
        base_dir: Base directory containing shards
        num_shards: Number of shards to generate paths for
        prefix: Filename prefix (default: "shard")

    Returns:
        List of shard paths: base_dir/shard-00000.array_record, ...
    """
    return [
        f"{base_dir}/{prefix}-{i:05d}.array_record"
        for i in range(num_shards)
    ]


def build_dataset_paths(
    array_record_path: str | list[str],
    dataset_type: Literal["coinrun", "minecraft_vpt", "latent"],
    index_max: int | None = None,
) -> list[str]:
    """Unified path builder for all dataset types.

    Args:
        array_record_path: Path or list of paths to ArrayRecord files/directories
        dataset_type: Type of dataset ("coinrun", "minecraft_vpt", "latent")
        index_max: For minecraft_vpt/latent, number of shards to load

    Returns:
        List of paths to ArrayRecord files

    Raises:
        ValueError: If index_max is not provided for minecraft_vpt or latent datasets
    """
    # Minecraft VPT and latent datasets use shard-based naming
    if dataset_type in ("minecraft_vpt", "latent"):
        if index_max is None or index_max <= 0:
            raise ValueError(
                f"index_max must be > 0 for {dataset_type} dataset, got {index_max}"
            )
        return generate_shard_paths(array_record_path, index_max)

    # CoinRun uses file discovery
    return discover_array_record_paths(array_record_path)
