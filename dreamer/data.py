import coinrun_data.dataloader as coinrun_loader
from .configs import DatasetConfig


def make_iterator(cfg: DatasetConfig):
    """
    Create data iterator for recorded trajectories in ArrayRecord format.

    Args:
        cfg: DatasetConfig with batch size, sequence length, and dataset path

    Returns:
        Grain dataloader that yields batches of (videos, actions, rewards)
    """
    return coinrun_loader.get_dataloader(
        array_record_paths=cfg.array_record_path,
        seq_len=cfg.T,
        global_batch_size=cfg.B,
        image_h=cfg.H,
        image_w=cfg.W,
        image_c=cfg.C,
        num_workers=8,
        print_filter_warnings=False,
        p_include_reward=cfg.p_include_reward,
    )
