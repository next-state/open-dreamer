from dataclasses import dataclass, field


# ---- Dataset Configs ----

@dataclass(frozen=False, unsafe_hash=True)
class DatasetConfig:
    """Configuration for dataset parameters.

    This config is shared across all experiments (tokenizer, dynamics, policy)
    to ensure consistent data loading.
    """
    name: str = "coinrun"

    # Batch and sequence dimensions
    B: int = 32  # batch size
    T: int = 64  # sequence length
    H: int = 64  # height
    W: int = 64  # width
    C: int = 3   # channels

    # Dataset path and action space
    action_dim: int = 1
    array_record_path: str = "datasets/coinrun_episodes/train"

    # Dataset normalization statistics (for pixel values in [0, 1])
    dataset_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    dataset_std: tuple[float, ...] = (0.288675, 0.288675, 0.288675)  # sqrt(1/12)

    # Reward-biased slicing probability (0.0 = disabled, 0.8 = 80% chance to include windows with nonzero reward)
    p_include_reward: float = 0.0

# ---- Model Configs ----

@dataclass(frozen=False)
class EncoderModelConfig:
    n_latents: int = 16
    d_bottleneck: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 2
    patch_size: int = 4
    depth: int = 8
    dropout_rate: float = 0.05
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    time_every: int = 4
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    dtype: str = "float32"
    param_dtype: str = "float32"
    
    dataset_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    dataset_std: tuple[float, ...] =(0.288675, 0.288675, 0.288675)  # sqrt(1/12)


@dataclass(frozen=False)
class DecoderModelConfig:
    d_bottleneck: int = 32  # Must match encoder's d_bottleneck
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 2
    n_latents: int = 16
    patch_size: int = 4
    d_patch: int = 48    # Will be computed from patch_size, C
    depth: int = 8
    dropout_rate: float = 0.05
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    time_every: int = 4
    dtype: str = "float32"
    param_dtype: str = "float32"
    H: int = 64
    W: int = 64
    
    dataset_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    dataset_std: tuple[float, ...] =(0.288675, 0.288675, 0.288675)  # sqrt(1/12)


@dataclass(frozen=False)
class TokenizerModelConfig:
    """Model configuration for tokenizer (encoder + decoder architecture)."""
    encoder: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    decoder: DecoderModelConfig = field(default_factory=DecoderModelConfig)


@dataclass(frozen=False, unsafe_hash=True)
class DynamicsModelConfig:
    d_model: int = 128
    d_bottleneck: int = 32
    action_dim: int = 16
    depth: int = 8
    n_heads: int = 4
    n_kv_heads: int = 2
    packing_factor: int = 2
    n_register: int = 4 # number of register tokens for dynamics
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    time_every: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    dtype: str = "float32"
    param_dtype: str = "float32"

    # schedule
    k_max: int = 8

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
    warmup_steps: int = 10_000
    wsd_decay_steps: int = 30_000
    max_steps: int = 1_000_000_000


@dataclass(frozen=False)
class CheckpointConfig:
    """Configuration for checkpointing."""
    max_to_keep: int = 5  # Maximum number of checkpoints to keep
    save_interval_steps: int = 10_000  # Save checkpoint every N steps
    max_steps: int = 1_000_000_000  # Maximum number of training steps


@dataclass(frozen=False)
class OptimizerConfig:
    """Configuration for optimizer."""
    # Optimizer type
    optimizer_type: str = "adamw"  # Currently only "adamw" supported

    # Optimizer hyperparameters (AdamW)
    b1: float = 0.9
    b2: float = 0.9
    weight_decay: float = 1e-4


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
    parallel_strategy: str = "data"  # Parallelization strategy: "data", "fsdp", or "tp"
    
    # Precision
    dtype: str = "bfloat16"
    param_dtype: str = "float32"


@dataclass(frozen=False)
class TokenizerConfig(BaseExperimentConfig):
    # Model architecture
    tokenizer: TokenizerModelConfig = field(default_factory=TokenizerModelConfig)

    # Training
    lpips_weight: float = 0.2
    lpips_frac: float = 0.5
    visualize_every: int = 10_000
    tokenizer_loss_type: str = "mae" # "mse" | "mae"

    # LR schedule
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=False)
class DynamicsConfig(BaseExperimentConfig):
    tokenizer_ckpt: str = ""  # checkpoint from train_tokenizer.py

    # Model
    dynamics: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)

    # Training
    max_steps: int = 50_000
    bootstrap_start: int = 5_000
    self_fraction: float = 0.25
    batch_size: int = 16

    # LR schedule
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    
    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely


@dataclass(frozen=False)
class HeadsConfig(BaseExperimentConfig):
    tokenizer_ckpt: str = ""  # checkpoint from train_tokenizer.py
    dynamics_ckpt: str = ""  # checkpoint from train_dynamics.py
    action_dim: int = 4  # number of categorical actions
    n_agent: int = 1  # number of agent tokens for dynamics

    # Training hyperparameters
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap masked out
    self_fraction: float = 0.25   # used once we pass bootstrap_start

    # Learning rate schedules (one per component)
    lr_schedule_policy: LRScheduleConfig = field(default_factory=lambda: LRScheduleConfig(lr=1e-4))
    lr_schedule_reward: LRScheduleConfig = field(default_factory=lambda: LRScheduleConfig(lr=1e-4))
    lr_schedule_dynamics: LRScheduleConfig = field(default_factory=lambda: LRScheduleConfig(lr=1e-5))

    # Optimizer config (shared across all components)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    dynamics_loss_weight: float = 0.1

    # Eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely

    # Multi-token prediction (MTP) settings
    L: int = 2                      # predict next L actions/rewards
    num_reward_bins: int = 101      # twohot bins for symexp rewards
    reward_log_low: float = -3.0    # log-space lower bound for reward bins (tune per dataset)
    reward_log_high: float = 3.0   # log-space upper bound for reward bins (tune per dataset)
    n_tasks: int = 128              # task-ID space for TaskEmbedder
    use_task_ids: bool = True       # True: discrete task IDs; False: vector embed
    
    # Loss weighting (to balance scales across different loss components)
    loss_weight_shortcut: float = 1.0    # weight for flow/bootstrap loss (MSE units)
    loss_weight_policy: float = 0.3      # weight for policy CE loss (nats)
    loss_weight_reward: float = 0.3      # weight for reward CE loss (nats)


@dataclass(frozen=False)
class RLConfig(BaseExperimentConfig):
    heads_ckpt: str = ""  # checkpoint from train_heads.py
    action_dim: int = 4

    # tokenizer / dynamics config
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 8
    n_heads: int = 4
    n_kv_heads: int = 2
    qk_norm_type: str | None = None
    rope_theta: float = 10000.0
    packing_factor: int = 2
    n_register: int = 4
    n_agent: int = 1
    agent_space_mode: str = "wm_agent"

    # schedule
    k_max: int = 8

    # train
    log_every: int = 5_000
    lr: float = 3e-4

    # eval media toggle
    write_video_every: int = 10_000
    visualize_every: int = 25_000

    # RL-specific
    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0
    reward_log_high: float = 3.0
    num_value_bins: int = 101
    n_tasks: int = 128
    use_task_ids: bool = True

    # RL hyperparameters
    gamma: float = 0.997
    lambda_: float = 0.95
    horizon: int = 32
    context_length: int = 16
    imagination_d: float = 1.0 / 4
    alpha: float = 0.5
    beta: float = 0.3

    # Evaluation
    eval_every: int = 50_000
    eval_episodes: int = 4
    eval_horizon: int = 32
    eval_batch_size: int = 4
    max_eval_examples_to_plot: int = 4
