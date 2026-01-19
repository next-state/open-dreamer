"""Scaling law utilities for compute-optimal training experiments.

This module provides:
1. Step computation functions for iso-FLOPs and compute-optimal training
2. ScalingContext class that encapsulates all scaling experiment logic

References:
- https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
- https://github.com/karpathy/nanochat
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def compute_max_steps(
    param_count: int,
    tokens_per_param: float,
    tokens_per_step: int,
) -> int:
    """Compute training steps from parameter count using fixed ratio.

    Following nanochat methodology: total_tokens = param_count × tokens_per_param

    Args:
        param_count: Total number of model parameters
        tokens_per_param: Target tokens per parameter ratio (e.g., 8.0)
        tokens_per_step: Tokens processed per training step (B × T)

    Returns:
        Number of training steps
    """
    total_tokens = param_count * tokens_per_param
    return int(total_tokens / tokens_per_step)


def compute_steps_for_flops_budget(
    total_flops: float,
    flops_per_step: int,
) -> int:
    """Compute training steps to achieve a fixed FLOPs budget.

    For iso-FLOPs experiments: train different model sizes for different
    durations such that total compute is constant.

    Args:
        total_flops: Target total FLOPs budget
        flops_per_step: FLOPs per training step for this model

    Returns:
        Number of training steps
    """
    return int(total_flops / flops_per_step)


@dataclass
class ScalingContext:
    """Encapsulates scaling experiment setup and tracking.

    This class handles all the overhead of scaling experiments:
    - Computing max_steps from FLOPs budget or tokens-per-param ratio
    - Tracking training time and final metrics
    - Writing results.csv only when in scaling mode
    """

    # Configuration
    cfg: Any
    param_count: int
    data_tokens_per_step: int
    total_tokens_per_step: int
    flops_per_step: float
    run_dir: Path
    is_active: bool

    # Tracking state (private)
    _start_time: float = field(default=0.0, repr=False)
    _final_loss: float = field(default=0.0, repr=False)
    _final_psnr: float = field(default=0.0, repr=False)
    _final_step: int = field(default=0, repr=False)

    @classmethod
    def create(
        cls,
        cfg: Any,
        param_count: int,
        flops_per_step: float,
        data_tokens_per_step: int,
        total_tokens_per_step: int,
        logger: Any,
        run_dir: Path,
    ) -> "ScalingContext":
        """Factory that applies scaling overrides to cfg if scaling mode is active.

        Args:
            cfg: Training config (must have scaling_flops_budget, scaling_tokens_per_param,
                 max_steps, lr_schedule.max_steps, ckpt.max_steps)
            param_count: Total model parameters
            flops_per_step: FLOPs per training step
            data_tokens_per_step: input tokens that come from the dataloader per step
            total_tokens_per_step: total tokens processed (including registers etc) per step
            logger: Logger instance (will set max_steps if scaling mode active)
            run_dir: Run directory (parent used for results.csv)

        Returns:
            Configured ScalingContext instance
        """
        is_active = cfg.scaling_flops_budget > 0 or cfg.scaling_tokens_per_param > 0

        if cfg.scaling_flops_budget > 0:
            # Iso-FLOPs mode: fixed compute budget
            steps_for_flops_budget = int(cfg.scaling_flops_budget / flops_per_step)
            cfg.max_steps = steps_for_flops_budget
            cfg.lr_schedule.max_steps = steps_for_flops_budget
            cfg.ckpt.max_steps = steps_for_flops_budget
            logger.max_steps = steps_for_flops_budget
            print(f"[IsoFLOPs] {cfg.scaling_flops_budget:.2e} FLOPs / {flops_per_step:.2e} per step = {steps_for_flops_budget:,} steps")

        elif cfg.scaling_tokens_per_param > 0:
            # Compute-optimal mode: fixed tokens per param ratio
            total_tokens = param_count * cfg.scaling_tokens_per_param
            steps_for_flops_budget = int(total_tokens / total_tokens_per_step)
            cfg.max_steps = steps_for_flops_budget
            cfg.lr_schedule.max_steps = steps_for_flops_budget
            cfg.ckpt.max_steps = steps_for_flops_budget
            logger.max_steps = steps_for_flops_budget
            total_tokens = param_count * cfg.scaling_tokens_per_param
            print(f"[Scaling] {param_count:,} params × {cfg.scaling_tokens_per_param} = {total_tokens:,.0f} tokens -> {steps_for_flops_budget:,} steps")

        return cls(
            cfg=cfg,
            param_count=param_count,
            data_tokens_per_step=data_tokens_per_step,
            total_tokens_per_step=total_tokens_per_step,
            flops_per_step=flops_per_step,
            run_dir=run_dir,
            is_active=is_active,
        )

    def start_training(self) -> None:
        """Call before the training loop starts."""
        self._start_time = time.time()

    def on_step(self, step: int, metrics: dict) -> None:
        """Call during logging to track final metrics for CSV output.

        Args:
            step: Current training step
            metrics: Metrics dict (looks for 'loss_total', 'flow_mse', 'psnr')
        """
        self._final_step = step
        self._final_loss = float(metrics.get("loss_total", metrics.get("flow_mse", 0.0)))
        self._final_psnr = float(metrics.get("psnr", 0.0))

    def get_step_metrics(self, step: int) -> dict:
        """Return extra metrics for W&B logging (always computed).

        Args:
            step: Current training step

        Returns:
            Dict with data_tokens_seen, total_tokens_seen, flops_spent
        """
        return {
            "data_tokens_seen": self.data_tokens_per_step * step,
            "total_tokens_seen": self.total_tokens_per_step * step,
            "flops_spent": self.flops_per_step * step,
        }

    def finalize(self) -> None:
        """Write results.csv line if in scaling mode. Call after training loop."""
        if not self.is_active:
            return

        train_elapsed = time.time() - self._start_time
        final_step = min(self._final_step, self.cfg.max_steps - 1)
        data_tokens_trained = self.data_tokens_per_step * final_step
        total_tokens_trained = self.total_tokens_per_step * final_step

        # Append to parent directory's results.csv (created by run_scaling.sh)
        results_csv = self.run_dir.parent / "results.csv"
        csv_line = (
            f"{self.cfg.run_name},{self.param_count},{self.data_tokens_per_step},"
            f"{self.total_tokens_per_step},{self.flops_per_step:.6e},"
            f"{self.cfg.scaling_flops_budget or 0},{final_step},"
            f"{data_tokens_trained},{total_tokens_trained},"
            f"{train_elapsed/3600:.4f},{self._final_loss:.6f},{self._final_psnr:.4f}\n"
        )
        with open(results_csv, "a") as f:
            f.write(csv_line)
