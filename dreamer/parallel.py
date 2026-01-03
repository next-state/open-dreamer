import jax
import math
from dataclasses import dataclass
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  attn: str | None = None
  data: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


# mesh_rules = MeshRules(
#   embed=None,  # Replicated
#   mlp='model',  # Model-parallel
#   attn='model',  # Model-parallel
#   data='data',  # Data-parallel
# )


def create_data_model_parallel(
    data_shards: int = 1,
    model_shards: int = 1
) -> tuple[Mesh, NamedSharding]:
    """Creates a data-model parallel mesh and data sharding."""
    devices = jax.devices()
    device_count = len(devices)

    axis_shapes = (data_shards, model_shards)
    axis_names = ('data', 'model')

    prod_axis_shapes = math.prod(axis_shapes)
    if device_count != prod_axis_shapes:
        raise ValueError(
            f"Device count ({device_count}) does not match the product of "
            f"axis shapes ({prod_axis_shapes})."
        )
    
    mesh = jax.make_mesh(axis_shapes, axis_names)
    data_sharding = NamedSharding(mesh, P('data', None))

    print(f"[parallel] Devices ({device_count}): {devices}")
    print(f"[parallel] Mesh axis shapes: {axis_shapes}, axis names: {axis_names}")
    
    return mesh, data_sharding
