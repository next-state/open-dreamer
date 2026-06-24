"""
Checkpoint bundles for saving and loading model groups.

Each bundle is a dataclass containing related models and their optimizers.
Bundles have `from_pretrained` class methods for loading checkpoints for inference
(without optimizers).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import ClassVar, Self

import jax
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import OmegaConf

from dreamer.configs import (
    CheckpointConfig,
    DynamicsModelConfig,
    PolicyHeadModelConfig,
    RewardHeadModelConfig,
    TaskEmbedderModelConfig,
    TokenizerModelConfig,
)
from dreamer.models import (
    Dynamics,
    PolicyHeadMTP,
    RewardHeadMTP,
    TaskEmbedder,
    Tokenizer,
)
from dreamer.training import RMSLossNormalizer
from dreamer.parallel import MeshRules
from jax.experimental import multihost_utils

from dreamer.utils import from_dict



class NoOpCheckpointManager(ocp.CheckpointManager):
    """No-op checkpoint manager that does nothing (used when max_to_keep == 0)."""
    def should_save(self, step: int) -> bool: return False


def build_checkpoint_manager(
        ckpt_cfg: CheckpointConfig,
        ckpt_dir: Path,
        item_names=("model_state", "optimizer_state", "rngs", "meta"),
    ) -> ocp.CheckpointManager:

    is_multihost = jax.process_count() > 1
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=ckpt_cfg.max_to_keep,
        save_interval_steps=ckpt_cfg.save_interval_steps,
        save_on_steps=[ckpt_cfg.max_steps - 1],  # always save at the end
        single_host_load_and_broadcast=is_multihost,
        enable_async_checkpointing=True,
        multiprocessing_options=ocp.options.MultiprocessingOptions(primary_host=0),
    )

    if ckpt_cfg.max_to_keep == 0:
        return NoOpCheckpointManager(ckpt_dir, options=checkpoint_options, item_names=item_names)
    return ocp.CheckpointManager(ckpt_dir, options=checkpoint_options, item_names=item_names)


class CheckpointBundle:
    """Base class for checkpoint bundles with save/restore/from_pretrained.

    Subclasses should be dataclasses that define:
    - `_model_registry`: ClassVar mapping field names to (ConfigClass, ModelClass) tuples
    - Fields for models (those in registry) and optional optimizers

    Example:
        @dataclass
        class TokenizerCheckpointBundle(CheckpointBundle):
            _model_registry: ClassVar[dict] = {
                "tokenizer": (TokenizerModelConfig, Tokenizer),
            }
            tokenizer: Tokenizer
            tokenizer_optimizer: Optional[nnx.Optimizer] = None
    """

    _model_registry: ClassVar[dict[str, tuple[type, type]]] = {}

    @classmethod
    def get_item_names(cls) -> tuple[str, ...]:
        """Get checkpoint item names for this bundle.

        Introspects the dataclass fields and adds standard items.

        Returns:
            Tuple of item names for checkpoint manager
        """
        if not is_dataclass(cls):
            raise TypeError(f"CheckpointBundle subclass must be a dataclass, got {cls}")

        item_names = [field.name for field in fields(cls)]
        item_names.extend(["rngs", "meta"])
        return tuple(item_names)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        mesh_rules: MeshRules,
        rngs: nnx.Rngs | None = None,
        model_names: set[str] | None = None,
    ) -> Self:
        """Load bundle from checkpoint (without optimizers).

        Uses the _model_registry to determine which fields are models and
        how to reconstruct them from saved configs.

        Args:
            checkpoint_path: Path to checkpoint directory
            mesh_rules: Mesh sharding rules
            rngs: Random number generators (default: Rngs(0))
            model_names: Optional subset of registry keys to load. If None, loads all.

        Returns:
            Bundle with loaded models, unloaded/optimizer fields set to None
        """
        if not cls._model_registry:
            raise NotImplementedError(
                f"{cls.__name__} must define _model_registry class variable"
            )

        if rngs is None:
            rngs = nnx.Rngs(0)
        checkpoint_path = str(Path(checkpoint_path).resolve())

        registry = cls._model_registry
        if model_names is not None:
            registry = {k: v for k, v in registry.items() if k in model_names}

        with ocp.CheckpointManager(checkpoint_path) as checkpoint_manager:
            step = checkpoint_manager.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

            # Load config from metadata
            meta_restored = checkpoint_manager.restore(
                step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
            )
            meta = meta_restored["meta"]

            # Initialize models from registry
            models = {}
            for field_name, (config_cls, model_cls) in registry.items():
                cfg = from_dict(config_cls, meta[field_name])
                models[field_name] = model_cls(cfg, mesh_rules=mesh_rules, rngs=rngs)

            # Restore weights
            restore_kwargs = {
                name: ocp.args.StandardRestore(nnx.state(model))
                for name, model in models.items()
            }
            restore_args = ocp.args.Composite(**restore_kwargs)
            restored = checkpoint_manager.restore(step, args=restore_args)

            # Update model weights
            for name, model in models.items():
                nnx.update(model, restored[name])

        # Build kwargs for dataclass constructor (models + None for optimizers)
        init_kwargs = dict(models)
        for field in fields(cls):
            if field.name not in models:
                init_kwargs[field.name] = None

        return cls(**init_kwargs)

    def restore(
        self,
        checkpoint_manager: ocp.CheckpointManager,
        rng: jax.Array,
    ) -> tuple[int, Self, jax.Array]:
        """Restore checkpoint state into this bundle (in-place update).

        Args:
            checkpoint_manager: Checkpoint manager
            rng: Random number generator state

        Returns:
            Tuple of (start_step, self, rng)
        """
        step = checkpoint_manager.latest_step()
        if step is None:
            print("No checkpoint found, starting from scratch.")
            return 0, self, rng

        # Build restore args dynamically by introspecting bundle fields
        restore_kwargs = {}

        for field in fields(self):
            field_value = getattr(self, field.name)
            if field_value is None:  # optional components (e.g. disabled qphi) — skip
                continue
            restore_kwargs[field.name] = ocp.args.StandardRestore(nnx.state(field_value))

        restore_kwargs["rngs"] = ocp.args.StandardRestore({"key": rng})

        restore_args = ocp.args.Composite(**restore_kwargs)
        restored = checkpoint_manager.restore(step, args=restore_args)

        # Update bundle fields in-place
        for field in fields(self):
            field_value = getattr(self, field.name)
            if field_value is None:
                continue
            nnx.update(field_value, restored[field.name])

        rng = restored["rngs"]["key"]
        print(f"Restored checkpoint from step {step}.")

        return step + 1, self, rng

    def maybe_save(
        self,
        checkpoint_manager: ocp.CheckpointManager,
        step: int,
        rngs: jax.Array,
    ) -> None:
        """Save checkpoint if checkpoint_manager.should_save(step).

        Args:
            checkpoint_manager: Checkpoint manager
            step: Current training step
            rngs: Random number generator state
        """
        if not checkpoint_manager.should_save(step):
            return

        # Build save args dynamically by introspecting bundle fields
        save_kwargs, meta = {}, {}

        for field in fields(self):
            field_value = getattr(self, field.name)
            if field_value is None:  # optional components (e.g. disabled qphi) — skip
                continue
            save_kwargs[field.name] = ocp.args.StandardSave(nnx.state(field_value))
            if hasattr(field_value, 'cfg'):
                cfg = field_value.cfg
                meta[field.name] = asdict(cfg) if is_dataclass(cfg) else OmegaConf.to_container(cfg, resolve=True)

        save_kwargs["rngs"] = ocp.args.StandardSave({'key': rngs})
        save_kwargs["meta"] = ocp.args.JsonSave(meta)

        save_args = ocp.args.Composite(**save_kwargs)
        checkpoint_manager.save(step, args=save_args)

@dataclass
class TokenizerCheckpointBundle(CheckpointBundle):
    """Bundle for tokenizer checkpoint save/restore."""

    _model_registry: ClassVar[dict[str, tuple[type, type]]] = {
        "tokenizer": (TokenizerModelConfig, Tokenizer),
        "tokenizer_ema": (TokenizerModelConfig, Tokenizer),
    }

    tokenizer: Tokenizer
    tokenizer_ema: Tokenizer
    tokenizer_optimizer: nnx.Optimizer | None = None


@dataclass
class DynamicsCheckpointBundle(CheckpointBundle):
    """Bundle for dynamics checkpoint save/restore.

    Note: tokenizer is included but frozen (not trained). Including it makes
    checkpoints self-contained for downstream usage.
    """

    _model_registry: ClassVar[dict[str, tuple[type, type]]] = {
        "dynamics": (DynamicsModelConfig, Dynamics),
        "dynamics_ema": (DynamicsModelConfig, Dynamics),
        "tokenizer": (TokenizerModelConfig, Tokenizer),
    }

    dynamics: Dynamics
    dynamics_ema: Dynamics
    tokenizer: Tokenizer
    dynamics_optimizer: nnx.Optimizer | None = None
    # Learned perturbation network (None when qphi is disabled). Deliberately NOT in
    # `_model_registry` so `from_pretrained` keeps loading pre-qphi checkpoints; it is
    # saved/restored only via the field-introspection path (which skips None fields).
    qphi: nnx.Module | None = None
    qphi_optimizer: nnx.Optimizer | None = None


@dataclass
class HeadsCheckpointBundle(CheckpointBundle):
    """Bundle for heads checkpoint save/restore.

    Note: tokenizer is included but frozen (not trained). Including it makes
    checkpoints self-contained for downstream usage (e.g., inference, visualization).
    """

    _model_registry: ClassVar[dict[str, tuple[type, type]]] = {
        "tokenizer": (TokenizerModelConfig, Tokenizer),
        "dynamics": (DynamicsModelConfig, Dynamics),
        "task_embedder": (TaskEmbedderModelConfig, TaskEmbedder),
        "policy_head": (PolicyHeadModelConfig, PolicyHeadMTP),
        "reward_head": (RewardHeadModelConfig, RewardHeadMTP),
    }

    tokenizer: Tokenizer
    dynamics: Dynamics
    task_embedder: TaskEmbedder
    policy_head: PolicyHeadMTP
    reward_head: RewardHeadMTP
    dynamics_optimizer: nnx.Optimizer | None = None
    task_embedder_optimizer: nnx.Optimizer | None = None
    policy_optimizer: nnx.Optimizer | None = None
    reward_optimizer: nnx.Optimizer | None = None
    loss_normalizer: RMSLossNormalizer | None = None
