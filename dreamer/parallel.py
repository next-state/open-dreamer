import jax
from dataclasses import dataclass
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing import Literal


@dataclass(unsafe_hash=True)
class MeshRules:
  embed: str | None = None
  mlp: str | None = None
  attn: str | None = None
  data: str | None = None
  seq: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


def build_parallel(strategy: Literal["data", "fsdp", "tp", "sp"]) -> tuple[Mesh, NamedSharding, MeshRules]:
    """Build parallelization setup based on strategy."""
    n = len(jax.devices())

    if strategy == "data":
        mesh = jax.make_mesh((n, 1), ('data', 'model'))
        sharding = NamedSharding(mesh, P('data', None))
        rules = MeshRules(embed=None, mlp='model', attn='model', data='data')

    elif strategy == "fsdp":
        mesh = jax.make_mesh((n, 1), ('data', 'model'))
        sharding = NamedSharding(mesh, P('data', None))
        rules = MeshRules(embed='data', mlp='data', attn='data', data='data')

    elif strategy == "tp":
        mesh = jax.make_mesh((1, n), ('data', 'model'))
        sharding = NamedSharding(mesh, P('data', None))
        rules = MeshRules(embed='model', mlp='model', attn='model', data=None)

    elif strategy == "sp":
        mesh = jax.make_mesh((1, n, 1), ('data', 'seq', 'model'))
        sharding = NamedSharding(mesh, P('data', 'seq', None, None))
        rules = MeshRules(embed=None, mlp='model', attn='model', data='data', seq='seq')

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'data', 'fsdp', 'tp', or 'sp'")

    print(f"[parallel] {strategy}: {dict(mesh.shape)}")
    return mesh, sharding, rules
