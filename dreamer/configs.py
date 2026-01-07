from dataclasses import dataclass, field


@dataclass(frozen=False, unsafe_hash=True)
class DatasetConfig:
    """Configuration for recorded trajectory datasets (ArrayRecord format)."""

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

@dataclass(frozen=False)
class EncoderConfig:
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
class DecoderConfig:
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
class TokenizerConfig:
    # IO / ckpt
    run_name: str
    ckpt_max_to_keep: int = 5
    ckpt_save_every: int = 10_000

    patch_size: int = 4

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None

    # dataset config
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # model parameters
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # training
    lr: float = 1e-4
    max_steps: int = 1_000_000_000
    log_every: int = 100
    lpips_weight: float = 0.2
    lpips_frac: float = 0.5
    visualize_every: int = 10_000
    tokenizer_loss_type: str = "mae" # "mse" | "mae"

    # precision
    dtype: str = "float32"
    param_dtype: str = "float32"

    # learning rate schedule
    # - "constant": use lr
    # - "wsd": warmup -> hold -> decay
    lr_schedule: str = "constant"
    init_lr: float = 0.0
    max_lr: float = 3e-4
    lr_end: float = 0.0
    warmup_steps: int = 10_000
    wsd_decay_steps: int = 30_000

    # logging
    log_gradients: bool = False

@dataclass(frozen=True)
class RLConfig:
    # IO / ckpt
    run_name: str
    bc_rew_ckpt: str  # checkpoint from train_bc_rew_heads.py
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None
    wandb_project: str | None = None

    # dataset config
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
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
    max_steps: int = 1_000_000_000
    log_every: int = 5_000
    lr: float = 3e-4

    # precision
    dtype: str = "float32"
    param_dtype: str = "float32"

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

@dataclass(frozen=False)
class DynamicsConfig:
    # IO / ckpt
    run_name: str
    tokenizer_ckpt: str
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # Nested
    dynamics: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None  # if None, uses default entity
    wandb_project: str | None = None  # if None, uses run_name as project

    # train
    max_steps: int = 50_000
    log_every: int = 5_000
    lr: float = 1e-4
    lr_schedule: str = "constant"
    init_lr: float = 0.0
    max_lr: float = 1e-4
    lr_end: float = 0.0
    warmup_steps: int = 5_000
    wsd_decay_steps: int = 20_000

    # precision
    dtype: str = "float32"
    param_dtype: str = "float32"

    # schedule
    bootstrap_start: int = 5_000
    self_fraction: float = 0.25
    batch_size: int = 16

    # eval media toggle
    write_video_every: int = 10_000  # set large to reduce IO, or 0 to disable entirely

@dataclass(frozen=True)
class BCRewConfig:
    # IO / ckpt
    run_name: str
    tokenizer_ckpt: str
    dynamics_ckpt: str
    ckpt_max_to_keep: int = 2
    ckpt_save_every: int = 10_000

    # wandb config
    use_wandb: bool = False
    wandb_entity: str | None = None  # if None, uses default entity
    wandb_project: str | None = None  # if None, uses run_name as project

    # dataset config
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    action_dim: int = 4  # number of categorical actions
    n_agent: int = 1  # number of agent tokens for dynamics

    # Training hyperparameters
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap masked out
    self_fraction: float = 0.25   # used once we pass bootstrap_start

    # Train
    max_steps: int = 1_000_000_000
    log_every: int = 5_000

    # precision
    dtype: str = "float32"
    param_dtype: str = "float32"
    
    # Learning rates
    lr_policy: float = 1e-4
    lr_reward: float = 1e-4
    lr_dynamics: float = 1e-5
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

@dataclass(frozen=True)
class EvalConfig:
    # Paths
    run_ckpt_dir: str                         # training run checkpoints dir to restore params from
    tokenizer_ckpt: str                       # tokenizer ckpt (for enc/dec)

    # dataset config
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    action_dim: int = 4

    # Tokenizer / dynamics
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
    k_max: int = 8

    # Heads
    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0    # log-space lower bound for reward bins (tune per dataset)
    reward_log_high: float = 3.0   # log-space upper bound for reward bins (tune per dataset)
    n_tasks: int = 128
    use_task_ids: bool = True

    # Sampler/eval
    ctx_length: int = 32
    horizon: int = 16
    schedule: str = "finest"  # "finest" or "shortcut"
    d: float | None = None    # e.g., 0.25 for shortcut
    tau_ctx: float = 1.0
    match_ctx_tau: bool = False

    # Visualization
    max_examples_to_plot: int = 4  # number of sequences to render as strips

    # Precision
    dtype: str = "float32"
    param_dtype: str = "float32"

    # Safety: ensure heads never see future actions when predicting next actions
    paranoid_no_leak: bool = True
