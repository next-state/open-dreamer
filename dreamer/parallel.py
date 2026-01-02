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
    data_sharding: NamedSharding
    replicated_sharding: NamedSharding
    device_count: int

    @classmethod
    def create(cls, batch_size: int) -> "ParallelContext":
        devices = jax.devices()
        device_count = len(devices)
        
        if batch_size % device_count != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by "
                f"device count ({device_count}) for data parallelism."
            )
        
        # Create device mesh
        mesh = Mesh(devices, axis_names=('data',))
        
        # Define sharding strategies
        data_sharding = NamedSharding(mesh, P('data', None))
        replicated_sharding = NamedSharding(mesh, P())
        
        ctx = cls(
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            device_count=device_count,
        )
        
        print(f"[parallel] Created context with {device_count} devices")
        print(f"[parallel] Global batch size: {batch_size}")
        print(f"[parallel] Per-device batch size: {batch_size // device_count}")
        print(f"[parallel] Devices: {devices}")
        
        return ctx
    
    def shard_data(self, tree: PyTree) -> PyTree:
        """Shard data along the batch dimension (Axis 0)."""
        return jax.tree.map(
            lambda x: jax.device_put(x, self.data_sharding), 
            tree
        )

    def replicate(self, tree: PyTree) -> PyTree:
        """Replicate scalars/params to all devices."""
        return jax.tree.map(
            lambda x: jax.device_put(x, self.replicated_sharding),
            tree
        )
    
    def to_host_scalar(self, tree: PyTree) -> PyTree:
        """
        Convert metrics/scalars from device to host, converting to Python scalars.
        """
        def _to_scalar(x):
            if isinstance(x, jax.Array):
                return x.item() if x.ndim == 0 else jax.device_get(x)
            return x
        return jax.tree.map(_to_scalar, tree)
