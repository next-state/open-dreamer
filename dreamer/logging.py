from typing import Any, Dict, Optional
import re

class MetricLogger:
    def __init__(
        self,
        use_wandb: bool,
        log_every: int,
        max_steps: Optional[int] = None,
        wandb_obj: Any = None,
    ):
        self.use_wandb = use_wandb
        self.log_every = log_every
        self.max_steps = max_steps
        self.wandb = wandb_obj

    def should_log(self, step: int) -> bool:
        if self.log_every <= 0:
            return False
        is_periodic = (step % self.log_every == 0)
        is_last = (self.max_steps is not None) and (step == self.max_steps)
        return is_periodic or is_last

    def log(
        self,
        step: int,
        metrics: Dict[str, Any],
        pbar: Any = None,
        prefix: str = "train/",
        float_fmt: str = ".4f",
        pbar_filter: Optional[str] = None,
    ):
        """
        Log metrics to tqdm and wandb (if enabled).

        Args:
            step: Current training step.
            metrics: Dictionary of metric names and values (can be JAX arrays).
            pbar: tqdm progress bar instance to update postfix.
            prefix: Prefix for wandb keys (default: "train/").
            float_fmt: Format string for floats in tqdm postfix (default: ".4f").
            pbar_filter: Optional regex string to filter which keys appear in progress bar.
                        If provided, only keys matching the pattern will be shown in pbar.
                        All metrics are still logged to wandb regardless of filter.
        """
        if not self.should_log(step):
            return

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

        if pbar is not None:
            postfix_data = {}
            filtered_metrics = clean_metrics
            
            # Apply filter if provided
            if pbar_filter is not None:
                pattern = re.compile(pbar_filter)
                filtered_metrics = {k: v for k, v in clean_metrics.items() if pattern.search(k)}
            
            for k, v in filtered_metrics.items():
                if isinstance(v, float):
                    postfix_data[k] = f"{v:{float_fmt}}"
                else:
                    postfix_data[k] = str(v)
            pbar.set_postfix(**postfix_data)

        if self.use_wandb and self.wandb is not None and getattr(self.wandb, "run", None) is not None:
            wandb_data = {f"{prefix}{k}": v for k, v in clean_metrics.items()}
            wandb_data[f"{prefix}step"] = step
            
            self.wandb.log(wandb_data, step=step)
