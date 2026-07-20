from dataclasses import dataclass, field
from typing import Literal


# ---- Dataset Configs ----

@dataclass(frozen=True, unsafe_hash=True)
class DataloaderConfig:
    """Configuration for dataloader parameters."""
    # Batch dimension
    B: int = 32  # batch size

    # Common parameters
    num_workers: int = 16
    prefetch_buffer_size: int = 10
    device_prefetch_buffer_size: int = 1

    # Dual iterator parameters (for alternating short/long sequences)
    short_T: int = 64
    long_T: int = 64
    long_ratio: float = 0.0
    start_step: int = 0

    dtype: str = "bfloat16"

@dataclass(frozen=False, unsafe_hash=True)
class DatasetConfig:
    """Configuration for dataset parameters.

    This config is shared across all experiments (tokenizer, dynamics, policy)
    to ensure consistent data loading.
    """
    name: str = "coinrun"

    dataloader_cfg: DataloaderConfig = field(default_factory=DataloaderConfig)
    # Frame dimensions
    H: int = 64  # frame height
    W: int = 64  # frame width
    C: int = 3   # channels
    padding_H: tuple[int,int] = field(default_factory=lambda: (0, 0))  # how much to pad each frame along the height axis
    padding_W: tuple[int,int] = field(default_factory=lambda: (0, 0))  # how much to pad each frame along the width axis

    patch_size: int = 8

    # Dataset path and action space
    num_binary_actions: int = 0
    categorical_action_dim: int = 0
    continuous_action_dim: int = 0
    mouse_repr: str = "categorical"  # "categorical" (mu-law bins, default) or "continuous" ([dx, dy])
    array_record_path: str = "datasets/coinrun_episodes/train"

    # For minecraft_vpt: number of shards to use (each shard = 1 episode)
    index_max: int = 0
    num_max_samples: int = -1

    # Dataset normalization statistics (for pixel values in [0, 1])
    dataset_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    dataset_std: tuple[float, ...] = (0.288675, 0.288675, 0.288675)  # sqrt(1/12)

    latent_mean: tuple[float, ...] | None = None
    latent_std: tuple[float, ...] | None = None

    # Reward-biased slicing probability (0.0 = disabled, 0.8 = 80% chance to include windows with nonzero reward)
    p_include_reward: float = 0.0

    # Data type: "video" (default) or "latent" (pre-tokenized)
    data_type: str = "video"


# ---- Model Configs ----

@dataclass(frozen=False)
class EncoderModelConfig:
    n_latents: int = 16
    d_bottleneck: int = 16
    depth: int = 4
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 1
    patch_size: int = 8
    dropout_rate: float = 0.05
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    time_every: int = 4
    time_layer_offset: int = 1
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    use_residual_lambdas: bool = False
    use_bias: bool = False
    use_rmsnorm_scale: bool = True
    use_seq_parallel: bool = False  # Enable sequence parallelism for time attention
    dtype: str = "float32"
    param_dtype: str = "float32"
    context_length: int | None = None  # sliding window size for temporal attention; None = full causal

    dataset_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    dataset_std: tuple[float, ...] =(0.288675, 0.288675, 0.288675)  # sqrt(1/12)


@dataclass(frozen=False)
class DecoderModelConfig:
    n_latents: int = 16
    d_bottleneck: int = 16  # Must match encoder's d_bottleneck
    depth: int = 4
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 1
    patch_size: int = 8
    d_patch: int = 192  # DatasetConfig.patch_size**2 * DatasetConfig.C
    dropout_rate: float = 0.05
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    time_every: int = 4
    time_layer_offset: int = 1
    use_residual_lambdas: bool = False
    use_bias: bool = False
    use_rmsnorm_scale: bool = True
    use_seq_parallel: bool = False  # Enable sequence parallelism for time attention
    dtype: str = "float32"
    param_dtype: str = "float32"
    context_length: int | None = None  # sliding window size for temporal attention; None = full causal
    H: int = 64  # sum(DatasetConfig.padding_H, DatasetConfig.H)
    W: int = 64  # sum(DatasetConfig.padding_W, DatasetConfig.W)

    dataset_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    dataset_std: tuple[float, ...] =(0.288675, 0.288675, 0.288675)  # sqrt(1/12)


@dataclass(frozen=False)
class TokenizerModelConfig:
    """Model configuration for tokenizer (encoder + decoder architecture)."""
    encoder: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    decoder: DecoderModelConfig = field(default_factory=DecoderModelConfig)


@dataclass(frozen=False, unsafe_hash=True)
class DynamicsModelConfig:
    d_bottleneck: int = 32
    depth: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 1
    packing_factor: int = 2
    n_register: int = 4 # number of register tokens for dynamics
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    time_every: int = 4
    time_layer_offset: int = 1
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    use_residual_lambdas: bool = False
    use_bias: bool = False
    use_rmsnorm_scale: bool = True
    use_seq_parallel: bool = False  # Enable sequence parallelism for time attention
    dtype: str = "float32"
    param_dtype: str = "float32"

    # schedule
    k_max: int = 8

    # attention window for sliding window attention (used when T > context_length)
    context_length: int = 192

    # action conditioning
    num_binary_actions: int = 0
    categorical_action_dim: int = 0
    continuous_action_dim: int = 0

    latent_mean: tuple[float, ...] | None = None
    latent_std: tuple[float, ...] | None = None


# ---- Experiment Configs ----

@dataclass(frozen=False)
class LRScheduleConfig:
    """Configuration for learning rate schedule."""
    # Schedule type
    # - "constant": constant learning rate (uses lr)
    # - "cos": warmup cosine decay
    # - "wsd": warmup -> hold -> decay (linear warmup, constant hold, linear decay)
    schedule_type: str = "constant"
    lr: float = 1e-4  # Used for constant schedule, or as peak/max_lr for other schedules
    init_lr: float = 0.0  # Starting LR for warmup schedules
    lr_end: float = 0.0  # Ending LR for decay schedules
    max_steps: int = 1_000_000_000
    warmup_ratio: float = 0.0  # e.g., 0.1 = 10% warmup
    decay_ratio: float = 0.0   # e.g., 0.2 = 20% decay (for wsd)


@dataclass(frozen=True, unsafe_hash=True)
class OptimalTransportConfig:
    """Configuration for minibatch optimal transport coupling."""
    enabled: bool = False
    epsilon: float | None = None
    max_iter: int = 50
    threshold: float = 1e-3
    lse_mode: bool = True
    pairing: str = "sample"  # "sample" | "argmax" | "barycentric"
    scale_cost: str = "mean"


@dataclass(frozen=False)
class CheckpointConfig:
    """Configuration for checkpointing."""
    max_to_keep: int = 5  # Maximum number of checkpoints to keep
    save_interval_steps: int = 10_000  # Save checkpoint every N steps
    max_steps: int = 1_000_000_000  # Maximum number of training steps


@dataclass(frozen=False)
class OptimizerConfig:
    """Configuration for optimizer."""
    # Optimizer type: "adamw", "laprop", or "muon"
    optimizer_type: str = "muon"
    weight_decay: float = 1e-4

    # AdamW hyperparameters (also used for non-matrix params when using Muon)
    adam_lr_ratio: float = 1.0      # adam_lr = muon_lr * adam_lr_ratio
    adam_b1: float = 0.9
    adam_b2: float = 0.9
    adam_eps: float = 1e-8

    # MuP scaling (scales AdamW LR by (d_model/mup_base_dim)^-0.5)
    mup_base_dim: int = 512         # Reference dimension for MuP scaling
    mup_scaling: bool = False       # Enable MuP LR scaling

    # Adaptive Gradient Clipping (AGC)
    agc_clipping: float = 0.0       # 0 = disabled; e.g. 0.3 clips at 30% of weight norm
    agc_eps: float = 1e-3           # Floor for weight norm in AGC

    # Muon-specific hyperparameters (for 2D matrix params, excluding embeddings)
    muon_beta: float = 0.95         # Momentum for Muon
    muon_ns_steps: int = 5          # Newton-Schulz iterations
    muon_nesterov: bool = True      # Use Nesterov momentum


@dataclass(frozen=False)
class LoggerConfig:
    """Configuration for experiment logging."""
    run_name: str = ""

    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None

    log_every: int = 100
    max_steps: int = 1_000_000_000
    log_gradients:  bool = False


@dataclass(frozen=False)
class BaseExperimentConfig:
    """Base configuration shared across all experiment types."""
    # IO
    run_name: str
    use_wandb: bool = False

    # Checkpoint
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Logger
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Training
    max_steps: int = 1_000_000_000
    log_every: int = 100
    seed: int = 0  # Random seed
    parallel_strategy: Literal["data", "fsdp", "tp", "sp"]  = "data"  # data | fsdp | tp | sp

    # Precision
    dtype: str = "bfloat16"
    param_dtype: str = "float32"

    # Scaling laws (set > 0 to enable auto-computed max_steps)
    scaling_tokens_per_param: float = 0.0  # e.g., 8.0 for compute-optimal training
    scaling_flops_budget: float = 0.0  # e.g., 1e17 for iso-FLOPs experiments


@dataclass(frozen=False)
class TokenizerConfig(BaseExperimentConfig):
    # Model architecture
    tokenizer: TokenizerModelConfig = field(default_factory=TokenizerModelConfig)

    # Training
    lpips_weight: float = 0.2
    lpips_frac: float = 0.5
    visualize_every: int = 10_000
    tokenizer_loss_type: str = "mae" # "mse" | "mae"

    # Finetuning: gradually reduce MAE masking while freezing encoder
    mae_finetune: bool = False
    mae_finetune_p_max_start: float | None = None
    mae_finetune_p_max_end: float | None = None

    # EMA model
    ema_decay: float = 0.999
    ema_dtype: str = "bfloat16"
    # LR schedule
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    #TODO: add dataset stats

@dataclass(frozen=False)
class DynamicsConfig(BaseExperimentConfig):
    tokenizer_ckpt: str = ""  # checkpoint from train_tokenizer.py

    # Model
    dynamics: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)

    # Training
    bootstrap_start: int = 5_000  # Number of start steps trained exclusively on flow-matching objective
    bootstrap_fraction: float = 0.25  # Fraction of batch used for bootstrap samples
    image_fraction: float = 0.3

    # Optimal transport coupling (minibatch OT)
    ot: OptimalTransportConfig = field(default_factory=OptimalTransportConfig)

    # Per-sample loss weighting (function of signal level sigma): "none" | "ramp" | "v_space"
    loss_weighting: str = "ramp"

    # LR schedule
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely

    # EMA model
    ema_decay: float = 0.999
    ema_dtype: str = "bfloat16"
