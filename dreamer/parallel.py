from dataclasses import dataclass
from typing import Any
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


PyTree = Any # Type alias for PyTree

@dataclass
class ParallelContext:
    """
    Manages mesh, sharding, and device placement for data parallelism.
    """
    mesh: Mesh
    batch_sharding: NamedSharding
    replicated_sharding: NamedSharding
    device_count: int

    @classmethod
    def create(cls, batch_size: int) -> "ParallelContext":
        """
        Create a ParallelContext for data parallelism.
        
        Args:
            batch_size: Global batch size (must be divisible by device count)
            
        Returns:
            ParallelContext configured for the available devices
            
        Raises:
            ValueError: If batch_size is not divisible by device count
        """
        devices = jax.devices()
        device_count = len(devices)
        
        if batch_size % device_count != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by "
                f"device count ({device_count}) for data parallelism."
            )
        
        per_device_batch_size = batch_size // device_count
        
        # Create device mesh
        mesh = Mesh(devices, axis_names=('batch',))
        
        # Define sharding strategies
        batch_sharding = NamedSharding(mesh, P('batch'))
        replicated_sharding = NamedSharding(mesh, P())
        
        ctx = cls(
            mesh=mesh,
            batch_sharding=batch_sharding,
            replicated_sharding=replicated_sharding,
            device_count=device_count,
        )
        
        print(f"[parallel] Created context with {device_count} devices")
        print(f"[parallel] Global batch size: {batch_size}")
        print(f"[parallel] Per-device batch size: {per_device_batch_size}")
        print(f"[parallel] Devices: {devices}")
        
        return ctx
    
    def shard_batch(self, tree: PyTree) -> PyTree:
        """
        Shard a pytree along the batch dimension across devices.
        
        Each device receives a slice of the batch (batch_size // device_count).
        
        Args:
            tree: Pytree with arrays having batch as first dimension
            
        Returns:
            Sharded pytree
        """
        return jax.tree.map(
            lambda x: jax.device_put(x, self.batch_sharding),
            tree
        )
    
    def replicate(self, tree: PyTree) -> PyTree:
        """
        Replicate a pytree across all devices.
        
        Use for model params, constants, and optimizer state.
        
        Args:
            tree: Pytree to replicate
            
        Returns:
            Replicated pytree
        """
        return jax.tree.map(
            lambda x: jax.device_put(x, self.replicated_sharding),
            tree
        )
    
    def split_keys(self, key: jax.Array, count: int) -> jax.Array:
        """        
        Generates one key per data sample (matching the batch size), then shards
        them to match the data layout. Each device automatically receives the slice
        of keys corresponding to its data slice.
        
        Args:
            key: Base RNG key
            count: Number of keys to generate (typically the global batch size)
            
        Returns:
            Sharded array of keys with shape (count,). Each device receives its
            corresponding slice automatically.
        """
        # Generate one key per sample in the global batch
        keys = jax.random.split(key, num=count)
        
        # Shard across devices
        return jax.device_put(keys, self.batch_sharding)
    
    def to_host_scalar(self, tree: PyTree) -> PyTree:
        """
        Convert metrics/scalars from device to host, converting to Python scalars.
        """
        def _to_scalar(x):
            if isinstance(x, jax.Array):
                return x.item() if x.ndim == 0 else jax.device_get(x)
            return x
        return jax.tree.map(_to_scalar, tree)
