"""Batching helpers for finite offline tokenization jobs."""

import grain


def make_tokenization_batch(batch_size: int) -> grain.transforms.Batch:
    """Create a batch operation that preserves the final partial batch."""
    return grain.transforms.Batch(
        batch_size=batch_size,
        drop_remainder=False,
    )
