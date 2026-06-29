import os
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


def _int_env(*names: str, default: int | None = None) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def maybe_init_distributed() -> None:
    """Initialize JAX multi-process runtime when launched across hosts."""
    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS")
    world = _int_env(
        "JAX_PROCESS_COUNT",
        "WORLD_SIZE",
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        default=1,
    )
    rank = _int_env(
        "JAX_PROCESS_INDEX",
        "RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        default=0,
    )

    slurm_env = "SLURM_JOB_ID" in os.environ
    ompi_env = "OMPI_MCA_orte_hnp_uri" in os.environ
    should_init = bool(coordinator) or (world is not None and world > 1) or slurm_env or ompi_env
    if not should_init:
        return

    is_initialized = getattr(jax.distributed, "is_initialized", None)
    if is_initialized is not None and is_initialized():
        return

    if coordinator:
        if world is None or world < 2:
            raise ValueError(
                "JAX_COORDINATOR_ADDRESS is set but process count is < 2. "
                "Set JAX_PROCESS_COUNT (or WORLD_SIZE)."
            )
        if rank is None:
            raise ValueError(
                "JAX_COORDINATOR_ADDRESS is set but process id is missing. "
                "Set JAX_PROCESS_INDEX (or RANK)."
            )
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=world,
            process_id=rank,
        )
    else:
        # Use launcher-provided metadata (e.g., Slurm/OpenMPI).
        jax.distributed.initialize()

    print(f"[distributed] initialized process {jax.process_index()}/{jax.process_count()}")


def build_parallel(strategy: Literal["data", "fsdp", "tp", "sp"]) -> tuple[Mesh, NamedSharding, MeshRules]:
    """Build parallelization setup based on strategy."""
    maybe_init_distributed()
    n = len(jax.devices())

    if strategy == "data":
        mesh = jax.make_mesh((n, 1), ('data', 'model'))
        sharding = NamedSharding(mesh, P('data', None))
        rules = MeshRules(embed=None, mlp='model', attn='model', data='data')

    elif strategy == "fsdp":
        mesh = jax.make_mesh((n, 1), ('data', 'model'))
        sharding = NamedSharding(mesh, P('data', None))
        # Keep embedding-like parameters replicated under FSDP.
        # Small vocab / token tables (e.g. binary action embeddings) are not
        # generally divisible by the data-axis mesh size, and only tensor
        # parallelism should shard embeddings.
        rules = MeshRules(embed=None, mlp='data', attn='data', data='data')

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
