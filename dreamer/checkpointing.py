"""
Checkpoint bundles for saving and loading model groups.

Each bundle is a dataclass containing related models and their optimizers.
Bundles have `from_pretrained` class methods for loading checkpoints for inference
(without optimizers).
"""
from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple, TypeVar

import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from dreamer.configs import (
    CheckpointConfig,
    DynamicsModelConfig,
    LRScheduleConfig,
    OptimizerConfig,
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
from dreamer.parallel import MeshRules
from dreamer.utils import from_dict



def build_checkpoint_manager(
        ckpt_cfg: CheckpointConfig,
        ckpt_dir: Path,
        item_names=("model_state", "optimizer_state", "train_dataloader_state", "rngs", "meta"),
    ) -> ocp.CheckpointManager:
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=ckpt_cfg.max_to_keep,
        save_interval_steps=ckpt_cfg.save_interval_steps,
        save_on_steps=[ckpt_cfg.max_steps - 1],  # always save at the end
    )

    return ocp.CheckpointManager(ckpt_dir, options=checkpoint_options, item_names=item_names)


BundleT = TypeVar('BundleT')


def get_bundle_item_names(bundle: BundleT) -> tuple[str, ...]:
    """Get checkpoint item names from a bundle dataclass.
    
    Introspects the bundle to generate the list of item names needed for
    build_checkpoint_manager.
    
    Args:
        bundle: Dataclass containing models and optimizers
        
    Returns:
        Tuple of item names for checkpoint manager
    """
    if not is_dataclass(bundle):
        raise TypeError(f"bundle must be a dataclass, got {type(bundle)}")
    
    item_names = [field.name for field in fields(bundle)]
    item_names.extend(["train_dataloader_state", "rngs", "meta"])
    
    return tuple(item_names)


def try_restore_bundle(
    checkpoint_manager: ocp.CheckpointManager,
    bundle: BundleT,
    train_iterator: grain.DataLoaderIterator,
    rng: jax.Array,
) -> tuple[int, BundleT, grain.DataLoaderIterator, jax.Array]:
    """Restore checkpoint bundle (generic for any dataclass).
    
    This function introspects the bundle dataclass to automatically save/restore
    all fields that are nnx.Module or nnx.Optimizer instances.
    
    Args:
        checkpoint_manager: Checkpoint manager
        bundle: Dataclass containing models and optimizers to restore
        train_iterator: Data iterator
        rng: Random number generator state
        
    Returns:
        Tuple of (start_step, bundle, train_iterator, rng)
        
    """
    if not is_dataclass(bundle):
        raise TypeError(f"bundle must be a dataclass, got {type(bundle)}")
    
    step = checkpoint_manager.latest_step()
    if step is None:
        print("No checkpoint found, starting from scratch.")
        return 0, bundle, train_iterator, rng
    
    # Build restore args dynamically by introspecting bundle fields
    restore_kwargs = {}
    
    for field in fields(bundle):
        field_value = getattr(bundle, field.name)
        restore_kwargs[field.name] = ocp.args.StandardRestore(nnx.state(field_value))  # type: ignore
    
    # Add iterator and rngs
    restore_kwargs["train_dataloader_state"] = grain.checkpoint.CheckpointRestore(train_iterator)  # type: ignore
    restore_kwargs["rngs"] = ocp.args.StandardRestore({"key": rng})  # type: ignore
    
    restore_args = ocp.args.Composite(**restore_kwargs)
    restored = checkpoint_manager.restore(step, args=restore_args)
    
    # Update bundle fields in-place
    for field in fields(bundle):
        field_value = getattr(bundle, field.name)
        nnx.update(field_value, restored[field.name])
    
    train_iterator = restored["train_dataloader_state"]
    rng = restored["rngs"]["key"]
    print(f"Restored checkpoint from step {step}.")
    
    return step + 1, bundle, train_iterator, rng


def maybe_save_bundle(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    bundle: BundleT,
    train_iterator: grain.DataLoaderIterator,
    rngs: jax.Array,
) -> None:
    """Save checkpoint bundle (generic for any dataclass).
    
    This function introspects the bundle dataclass to automatically save/restore
    all fields that are nnx.Module or nnx.Optimizer instances.
    
    Args:
        checkpoint_manager: Checkpoint manager
        step: Current training step
        bundle: Dataclass containing models and optimizers to save
        train_iterator: Data iterator
        rngs: Random number generator state
        meta: Metadata dictionary
    """
    if not is_dataclass(bundle):
        raise TypeError(f"bundle must be a dataclass, got {type(bundle)}")
    
    if not checkpoint_manager.should_save(step):
        return
    
    # Build save args dynamically by introspecting bundle fields
    save_kwargs, meta = {}, {}
    
    for field in fields(bundle):
        field_value = getattr(bundle, field.name)
        save_kwargs[field.name] = ocp.args.StandardSave(nnx.state(field_value))  # type: ignore
        if hasattr(field_value, 'cfg'):
            cfg = field_value.cfg
            meta[field.name] = asdict(cfg) if is_dataclass(cfg) else OmegaConf.to_container(cfg, resolve=True)
               
    
    # Add iterator, rngs, and meta
    save_kwargs["train_dataloader_state"] = grain.checkpoint.CheckpointSave(train_iterator)  # type: ignore
    save_kwargs["rngs"] = ocp.args.StandardSave({'key': rngs})  # type: ignore
    save_kwargs["meta"] = ocp.args.JsonSave(meta)  # type: ignore
    
    save_args = ocp.args.Composite(**save_kwargs)
    checkpoint_manager.save(step, args=save_args)

@dataclass
class TokenizerCheckpointBundle:
    """Bundle for tokenizer checkpoint save/restore."""

    tokenizer: Tokenizer
    tokenizer_optimizer: Optional[nnx.Optimizer] = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        mesh_rules: MeshRules,
        rngs: Optional[nnx.Rngs] = None,
    ) -> "TokenizerCheckpointBundle":
        """Load tokenizer bundle from checkpoint (without optimizer).

        Args:
            checkpoint_path: Path to checkpoint directory
            mesh_rules: Mesh sharding rules
            rngs: Random number generators (default: Rngs(0))

        Returns:
            TokenizerCheckpointBundle with loaded tokenizer, optimizer=None
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        checkpoint_path = str(Path(checkpoint_path).resolve())

        with ocp.CheckpointManager(checkpoint_path) as checkpoint_manager:
            step = checkpoint_manager.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

            # Load config from metadata
            meta_restored = checkpoint_manager.restore(
                step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
            )
            cfg = from_dict(TokenizerModelConfig, meta_restored["meta"]["tokenizer"])

            # Initialize tokenizer
            tokenizer = Tokenizer(cfg, mesh_rules=mesh_rules, rngs=rngs)

            # Restore weights
            restore_args = ocp.args.Composite(
                tokenizer=ocp.args.StandardRestore(nnx.state(tokenizer))
            )
            restored = checkpoint_manager.restore(step, args=restore_args)
            nnx.update(tokenizer, restored["tokenizer"])

        return cls(tokenizer=tokenizer, tokenizer_optimizer=None)


@dataclass
class DynamicsCheckpointBundle:
    """Bundle for dynamics checkpoint save/restore.

    Note: tokenizer is included but frozen (not trained). Including it makes
    checkpoints self-contained for downstream usage.
    """

    dynamics: Dynamics
    tokenizer: Tokenizer
    dynamics_optimizer: Optional[nnx.Optimizer] = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        mesh_rules: MeshRules,
        rngs: Optional[nnx.Rngs] = None,
    ) -> "DynamicsCheckpointBundle":
        """Load dynamics bundle from checkpoint (without optimizer).

        Args:
            checkpoint_path: Path to checkpoint directory
            mesh_rules: Mesh sharding rules
            rngs: Random number generators (default: Rngs(0))

        Returns:
            DynamicsCheckpointBundle with loaded dynamics and tokenizer, optimizer=None
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        checkpoint_path = str(Path(checkpoint_path).resolve())

        with ocp.CheckpointManager(checkpoint_path) as checkpoint_manager:
            step = checkpoint_manager.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

            # Load config from metadata
            meta_restored = checkpoint_manager.restore(
                step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
            )
            cfg = meta_restored["meta"]

            # Reconstruct configs
            tokenizer_cfg = from_dict(TokenizerModelConfig, cfg["tokenizer"])
            dynamics_cfg = from_dict(DynamicsModelConfig, cfg["dynamics"])

            # Initialize models
            tokenizer = Tokenizer(tokenizer_cfg, mesh_rules=mesh_rules, rngs=rngs)
            dynamics = Dynamics(dynamics_cfg, mesh_rules=mesh_rules, rngs=rngs)

            # Restore weights
            restore_args = ocp.args.Composite(
                tokenizer=ocp.args.StandardRestore(nnx.state(tokenizer)),
                dynamics=ocp.args.StandardRestore(nnx.state(dynamics)),
            )
            restored = checkpoint_manager.restore(step, args=restore_args)
            nnx.update(tokenizer, restored["tokenizer"])
            nnx.update(dynamics, restored["dynamics"])

        return cls(dynamics=dynamics, tokenizer=tokenizer, dynamics_optimizer=None)


@dataclass
class HeadsCheckpointBundle:
    """Bundle for heads checkpoint save/restore.

    Note: tokenizer is included but frozen (not trained). Including it makes
    checkpoints self-contained for downstream usage (e.g., inference, visualization).
    """

    tokenizer: Tokenizer
    dynamics: Dynamics
    task_embedder: TaskEmbedder
    policy_head: PolicyHeadMTP
    reward_head: RewardHeadMTP
    dynamics_optimizer: Optional[nnx.Optimizer] = None
    task_embedder_optimizer: Optional[nnx.Optimizer] = None
    policy_optimizer: Optional[nnx.Optimizer] = None
    reward_optimizer: Optional[nnx.Optimizer] = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        mesh_rules: MeshRules,
        rngs: Optional[nnx.Rngs] = None,
    ) -> "HeadsCheckpointBundle":
        """Load heads bundle from checkpoint (without optimizers).

        Args:
            checkpoint_path: Path to checkpoint directory
            mesh_rules: Mesh sharding rules
            rngs: Random number generators (default: Rngs(0))

        Returns:
            HeadsCheckpointBundle with all models loaded, optimizers=None
        """
        if rngs is None:
            rngs = nnx.Rngs(0)
        checkpoint_path = str(Path(checkpoint_path).resolve())

        with ocp.CheckpointManager(checkpoint_path) as checkpoint_manager:
            step = checkpoint_manager.latest_step()
            if step is None:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

            # Load config from metadata (uses class names as keys)
            meta_restored = checkpoint_manager.restore(
                step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
            )
            meta = meta_restored["meta"]

            # Reconstruct configs using class names as keys
            tokenizer_cfg = from_dict(TokenizerModelConfig, meta["tokenizer"])
            dynamics_cfg = from_dict(DynamicsModelConfig, meta["dynamics"])
            task_embedder_cfg = from_dict(TaskEmbedderModelConfig, meta["task_embedder"])
            policy_head_cfg = from_dict(PolicyHeadModelConfig, meta["policy_head"])
            reward_head_cfg = from_dict(RewardHeadModelConfig, meta["reward_head"])

            # Split rngs for each model
            tok_key, dyn_key, task_key, pol_key, rew_key = [
                nnx.Rngs(i) for i in range(5)
            ]

            # Initialize all models with their configs
            tokenizer = Tokenizer(tokenizer_cfg, mesh_rules=mesh_rules, rngs=tok_key)
            dynamics = Dynamics(dynamics_cfg, mesh_rules=mesh_rules, rngs=dyn_key)
            task_embedder = TaskEmbedder(task_embedder_cfg, mesh_rules=mesh_rules, rngs=task_key)
            policy_head = PolicyHeadMTP(policy_head_cfg, mesh_rules=mesh_rules, rngs=pol_key)
            reward_head = RewardHeadMTP(reward_head_cfg, mesh_rules=mesh_rules, rngs=rew_key)

            # Restore weights for all models
            restore_args = ocp.args.Composite(
                tokenizer=ocp.args.StandardRestore(nnx.state(tokenizer)),
                dynamics=ocp.args.StandardRestore(nnx.state(dynamics)),
                task_embedder=ocp.args.StandardRestore(nnx.state(task_embedder)),
                policy_head=ocp.args.StandardRestore(nnx.state(policy_head)),
                reward_head=ocp.args.StandardRestore(nnx.state(reward_head)),
            )
            restored = checkpoint_manager.restore(step, args=restore_args)
            nnx.update(tokenizer, restored["tokenizer"])
            nnx.update(dynamics, restored["dynamics"])
            nnx.update(task_embedder, restored["task_embedder"])
            nnx.update(policy_head, restored["policy_head"])
            nnx.update(reward_head, restored["reward_head"])

        return cls(
            tokenizer=tokenizer,
            dynamics=dynamics,
            task_embedder=task_embedder,
            policy_head=policy_head,
            reward_head=reward_head,
            dynamics_optimizer=None,
            task_embedder_optimizer=None,
            policy_optimizer=None,
            reward_optimizer=None,
        )
