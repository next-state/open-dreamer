from typing import Any, Dict, Optional
from pathlib import Path
from typing import Literal
import os
import re

import jax

from dreamer.configs import LoggerConfig


class Logger:
    """
    Base logger with no-op default implementations.

    Can be used directly for no logging (only tqdm updates), or subclassed
    for specific backends like WandbLogger.
    """
    def __init__(self, logger_cfg: LoggerConfig, max_steps: Optional[int] = None):
        self.log_every = logger_cfg.log_every
        self.max_steps = max_steps
        self._initialized = False

    def should_log(self, step: int) -> bool:
        if self.log_every <= 0:
            return False
        is_periodic = (step % self.log_every == 0)
        is_last = (self.max_steps is not None) and (step == self.max_steps - 1)
        return is_periodic or is_last

    def __enter__(self):
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._initialized = False
        return False

    def log_metrics(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def log_video(self, *args, **kwargs):
        pass

    def log(self, step: int, metrics: Dict[str, Any], pbar: Any = None, prefix: str = "train/", float_fmt: str = ".4f", pbar_filter: Optional[str] = None):
        if not self.should_log(step):
            return

        # Convert JAX arrays to Python scalars
        clean_metrics = self._convert_metrics(metrics)

        # Update tqdm progress bar (all loggers support this)
        if pbar is not None:
            self._update_pbar(pbar, clean_metrics, float_fmt, pbar_filter)

        # Log to backend
        self.log_metrics(step, clean_metrics, prefix)

    def _convert_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JAX arrays and other types to Python scalars."""
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                v = v.item()

            if isinstance(v, (float, int)):
                clean_metrics[k] = v
            else:
                try:
                    clean_metrics[k] = float(v)
                except (ValueError, TypeError):
                    clean_metrics[k] = v
        return clean_metrics

    def _update_pbar(
        self,
        pbar: Any,
        metrics: Dict[str, Any],
        float_fmt: str,
        pbar_filter: Optional[str],
    ):
        """Update tqdm progress bar with filtered metrics."""
        filtered_metrics = metrics

        # Apply regex filter if provided
        if pbar_filter is not None:
            pattern = re.compile(pbar_filter)
            filtered_metrics = {k: v for k, v in metrics.items() if pattern.search(k)}

        # Format values for display
        postfix_data = {}
        for k, v in filtered_metrics.items():
            if isinstance(v, float):
                postfix_data[k] = f"{v:{float_fmt}}"
            else:
                postfix_data[k] = str(v)

        pbar.set_postfix(**postfix_data)


class WandbLogger(Logger):
    def __init__(
        self,
        logger_cfg: LoggerConfig,
        config: Any = None,
        dir: Optional[str] = None,
    ):
        super().__init__(logger_cfg=logger_cfg)
        self.entity = logger_cfg.wandb_entity
        self.project = logger_cfg.wandb_project
        self.name = logger_cfg.run_name
        self.config = config
        self.dir = dir
        self._run = None

    def __enter__(self):
        import wandb
        self._run = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            config=self.config,
            dir=self.dir,
        )
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._run is not None:
            import wandb
            wandb.finish()
        return False

    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = "train/"):
        import wandb
        wandb_data = {f"{prefix}{k}": v for k, v in metrics.items()}
        wandb_data[f"{prefix}step"] = step
        wandb.log(wandb_data, step=step)

    def log_image(self, step: int, key: str, image: Any, caption: Optional[str] = None):
        import wandb
        import numpy as np
        if hasattr(image, '__array__'):
            image = np.asarray(image)
        wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_video(self, step: int, key: str, video_path: Path, format: Literal["gif", "mp4", "webm", "ogg"] | None = "mp4", **kwargs):
        import wandb
        wandb.log({key: wandb.Video(str(video_path), format=format, **kwargs)}, step=step)


def build_logger(
    logger_cfg: LoggerConfig,
    config: Any = None,
    dir: Optional[str] = None,
) -> Logger:
    def _is_primary_process() -> bool:
        if jax.distributed.is_initialized():
            return jax.process_index() == 0
        rank_env = (
            os.environ.get("JAX_PROCESS_INDEX")
            or os.environ.get("RANK")
            or os.environ.get("SLURM_PROCID")
            or os.environ.get("OMPI_COMM_WORLD_RANK")
            or "0"
        )
        return int(rank_env) == 0

    use_wandb = logger_cfg.use_wandb and _is_primary_process()
    if use_wandb:
        return WandbLogger(
            logger_cfg=logger_cfg,
            config=config,
            dir=dir,
        )
    else:
        return Logger(logger_cfg=logger_cfg)
