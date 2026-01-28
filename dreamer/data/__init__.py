"""Data loading and processing for Dreamer.

This module provides unified data loading utilities for CoinRun, Minecraft VPT,
and pre-tokenized latent datasets.
"""

# Public API - core functions
from .data import make_iterator, make_dual_iterator

# Transforms (for advanced usage)
from .transforms import (
    EpisodeLengthFilter,
    ProcessEpisodeAndSlice,
    ProcessMinecraftEpisodeAndSlice,
    ProcessLatentAndSlice,
    CreateActions,
)

# Utilities (for dataset generation)
from .shard_writer import ShardWriter
from . import serialization
from . import path_utils

__all__ = [
    # Core API
    "make_iterator",
    "make_dual_iterator",
    # Transforms
    "EpisodeLengthFilter",
    "ProcessEpisodeAndSlice",
    "ProcessMinecraftEpisodeAndSlice",
    "ProcessLatentAndSlice",
    "CreateActions",
    # Utilities
    "ShardWriter",
    "serialization",
    "path_utils",
]
