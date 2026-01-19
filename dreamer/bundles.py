"""
Checkpoint bundles for saving and loading model groups.

Each bundle is a dataclass containing related models and their optimizers.
Bundles have `from_pretrained` class methods for loading checkpoints for inference
(without optimizers).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import orbax.checkpoint as ocp
from flax import nnx

from dreamer.configs import (
    DynamicsModelConfig,
    TokenizerModelConfig,
    TaskEmbedderModelConfig,
    PolicyHeadModelConfig,
    RewardHeadModelConfig,
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
