"""Serialization utilities for ArrayRecord data.

Provides unified serialization/deserialization for different data formats:
- Pickle-based: CoinRun and Minecraft VPT raw video data
- Msgpack-based: Pre-tokenized latent data with efficient numpy array encoding
"""

import msgpack
import numpy as np
import pickle
from typing import Any


# ==============================================================================
# Pickle-based serialization (CoinRun, Minecraft VPT raw video)
# ==============================================================================

def serialize_pickle_record(record: dict) -> bytes:
    """Serialize record using pickle.

    Used for CoinRun and Minecraft VPT datasets containing raw video bytes.

    Args:
        record: Dictionary with episode data (raw_video, actions, rewards, etc.)

    Returns:
        Pickled bytes
    """
    return pickle.dumps(record)


def deserialize_pickle_record(data: bytes) -> dict:
    """Deserialize record from pickle format.

    Args:
        data: Pickled bytes

    Returns:
        Dictionary with episode data
    """
    return pickle.loads(data)


# ==============================================================================
# Msgpack-based serialization (latent data)
# ==============================================================================

def _encode_value(value: Any) -> Any:
    """Encode a single value, handling arrays and nested dicts."""
    if isinstance(value, np.ndarray):
        return {
            "_type": "ndarray",
            "data": value.tobytes(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    elif isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    else:
        return value


def _decode_value(value: Any) -> Any:
    """Decode a single value, handling arrays and nested dicts."""
    if isinstance(value, dict):
        if value.get("_type") == "ndarray":
            shape = tuple(value["shape"])
            dtype = value["dtype"]
            return np.frombuffer(value["data"], dtype=dtype).reshape(shape)
        else:
            return {k: _decode_value(v) for k, v in value.items()}
    return value


def serialize_msgpack_record(record: dict) -> bytes:
    """Serialize record using msgpack (no pickle).

    Arrays are serialized via tobytes() with shape/dtype metadata.
    Nested dicts are recursively encoded.

    Used for pre-tokenized latent datasets for efficient storage.

    Args:
        record: Dictionary with latent data (latents, actions, etc.)

    Returns:
        Msgpack-encoded bytes
    """
    encoded = {k: _encode_value(v) for k, v in record.items()}
    return msgpack.packb(encoded, use_bin_type=True)


def deserialize_msgpack_record(data: bytes) -> dict:
    """Deserialize record from msgpack format.

    Reconstructs numpy arrays from bytes + shape/dtype metadata.
    Nested dicts are recursively decoded.

    Args:
        data: Msgpack-encoded bytes

    Returns:
        Dictionary with latent data
    """
    encoded = msgpack.unpackb(data, raw=False)
    return {k: _decode_value(v) for k, v in encoded.items()}


# Backward compatibility alias
deserialize_latent_record = deserialize_msgpack_record
