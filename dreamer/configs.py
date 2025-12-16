from dataclasses import dataclass, field
import math
from typing import Any, List, Optional


# --- Dataset Configs ---

@dataclass
class DatasetConfig:
    name: str = "bouncing_square"
    B: int = 32
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    
    pixels_per_step: int = 2
    size_min: int = 6
    size_max: int = 14
    hold_min: int = 4
    hold_max: int = 9
    
    diversify_data: bool = True
    
    dataset_mean: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    dataset_std: List[float] = field(default_factory=lambda: [0.288675, 0.288675, 0.288675])


# --- Model Configs ---

@dataclass
class EncoderModelConfig:
    d_model: int = 64
    n_latents: int = 16
    n_patches: int = field(init=False, default=0) # derived
    
    n_heads: int = 2
    n_kv_heads: int = 1
    depth: int = 2
    
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    use_rope: bool = False
    rope_theta: float = 10000.0
    
    time_every: int = 4
    mae_p_min: float = 0.0
    mae_p_max: float = 0.9
    
    d_bottleneck: int = 32

    def __post_init__(self):
         # Validation?
         pass

    def compute_derived(self, H: int, W: int, patch_size: int):
        # n_patches = (H//patch_size) * (W//patch_size)
        self.n_patches = (H // patch_size) * (W // patch_size)

@dataclass
class DecoderModelConfig:
    d_model: int = 64
    n_heads: int = 2
    n_kv_heads: int = 1
    depth: int = 2
    n_latents: int = 16 # Must match encoder
    
    n_latents: int = 16 # Must match encoder
    
    n_patches: int = field(init=False, default=0)
    d_patch: int = field(init=False, default=0)
    
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    use_rope: bool = False
    rope_theta: float = 10000.0
    
    time_every: int = 4

    def compute_derived(self, H: int, W: int, C: int, patch_size: int):
        self.n_patches = (H // patch_size) * (W // patch_size)
        self.d_patch = patch_size * patch_size * C

@dataclass
class TokenizerModelConfig:
    encoder: EncoderModelConfig = field(default_factory=EncoderModelConfig)
    decoder: DecoderModelConfig = field(default_factory=DecoderModelConfig)
    patch_size: int = 4

    def __post_init__(self):
        # Ensure consistency
        if self.decoder.n_latents != self.encoder.n_latents:
             raise ValueError("Encoder and Decoder must have same n_latents")

    def compute_derived(self, dataset_cfg: DatasetConfig):
        self.encoder.compute_derived(dataset_cfg.H, dataset_cfg.W, self.patch_size)
        self.decoder.compute_derived(dataset_cfg.H, dataset_cfg.W, dataset_cfg.C, self.patch_size)


@dataclass
class DynamicsModelConfig:
    d_model: int = 128
    d_bottleneck: int = 32
    d_spatial: int = field(init=False, default=0)
    n_spatial: int = field(init=False, default=0)
    
    n_register: int = 4
    n_agent: int = 1
    
    n_heads: int = 4
    n_kv_heads: int = 2
    depth: int = 4
    
    k_max: int = 8
    
    dropout_rate: float = 0.0
    qk_norm_type: str | None = None
    use_rope: bool = False
    rope_theta: float = 10000.0
    
    mlp_ratio: float = 4.0
    time_every: int = 4
    
    packing_factor: int = 2
    
    def compute_derived(self, encoder_cfg: EncoderModelConfig):
        self.d_bottleneck = encoder_cfg.d_bottleneck
        self.d_spatial = self.d_bottleneck * self.packing_factor
        
        if encoder_cfg.n_latents % self.packing_factor != 0:
            raise ValueError(f"Encoder n_latents {encoder_cfg.n_latents} not divisible by packing {self.packing_factor}")
        self.n_spatial = encoder_cfg.n_latents // self.packing_factor


@dataclass
class TaskEmbedderModelConfig:
    d_model: int = 128
    n_agent: int = 1
    use_ids: bool = True
    n_tasks: int = 128
    d_task: int = 64


@dataclass
class PolicyHeadModelConfig:
    d_model: int = 128
    action_dim: int = 4
    L: int = 2
    kind: str = "categorical"
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False


@dataclass
class RewardHeadModelConfig:
    d_model: int = 128
    L: int = 2
    num_bins: int = 101
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    log_low: float = -3.0
    log_high: float = 3.0


@dataclass
class ValueHeadModelConfig:
    d_model: int = 128
    num_bins: int = 101
    mlp_ratio: float = 2.0
    dropout_rate: float = 0.0
    swiglu: bool = True
    parity_2over3: bool = False
    log_low: float = -3.0
    log_high: float = 3.0


@dataclass
class BCRewModelConfig:
    task_embedder: TaskEmbedderModelConfig = field(default_factory=TaskEmbedderModelConfig)
    policy_head: PolicyHeadModelConfig = field(default_factory=PolicyHeadModelConfig)
    reward_head: RewardHeadModelConfig = field(default_factory=RewardHeadModelConfig)
    
    def compute_derived(self, dynamics_cfg: DynamicsModelConfig):
        """Set d_model and n_agent from loaded dynamics config to ensure consistency."""
        self.task_embedder.d_model = dynamics_cfg.d_model
        self.task_embedder.n_agent = dynamics_cfg.n_agent
        self.policy_head.d_model = dynamics_cfg.d_model
        self.reward_head.d_model = dynamics_cfg.d_model


@dataclass
class PolicyModelConfig:
    policy_head: PolicyHeadModelConfig = field(default_factory=PolicyHeadModelConfig)
    value_head: ValueHeadModelConfig = field(default_factory=ValueHeadModelConfig)
    
    def compute_derived(self, dynamics_cfg: DynamicsModelConfig):
        """Set d_model from loaded dynamics config to ensure consistency."""
        self.policy_head.d_model = dynamics_cfg.d_model
        self.value_head.d_model = dynamics_cfg.d_model

# --- Experiment Configs ---

@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    max_steps: int = 1_000_000_000

@dataclass
class CheckpointConfig:
    max_to_keep: int = 2
    save_every: int = 10_000

@dataclass
class WandbConfig:
    enabled: bool = False
    entity: Optional[str] = None
    project: Optional[str] = None

@dataclass
class ExperimentConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    log_every: int = 100

@dataclass
class TokenizerExperimentConfig(ExperimentConfig):
    lpips_weight: float = 0.2
    lpips_frac: float = 0.5
    visualize_every: int = 10_000

@dataclass
class DynamicsExperimentConfig(ExperimentConfig):
    tokenizer_ckpt_path: str = "logs/tokenizer"
    
    # Schedule
    k_max: int = 8
    bootstrap_start: int = 5_000
    self_fraction: float = 0.25
    
    write_video_every: int = 5_000


@dataclass
class BCRewExperimentConfig(ExperimentConfig):
    tokenizer_ckpt_path: str = "logs/tokenizer"
    dynamics_ckpt_path: str = "logs/dynamics"
    
    # Schedule
    k_max: int = 8
    bootstrap_start: int = 5_000
    self_fraction: float = 0.25
    
    # Head settings
    action_dim: int = 4
    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0
    reward_log_high: float = 3.0
    n_tasks: int = 128
    use_task_ids: bool = True
    
    # Loss weights
    loss_weight_shortcut: float = 1.0
    loss_weight_policy: float = 1.0
    loss_weight_reward: float = 1.0
    
    write_video_every: int = 10_000


@dataclass
class PolicyExperimentConfig(ExperimentConfig):
    tokenizer_ckpt_path: str = "logs/tokenizer"
    bc_rew_ckpt_path: str = "logs/bc_rew"
    
    # Head settings
    action_dim: int = 4
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
    imagination_d: float = 0.25
    alpha: float = 0.5
    beta: float = 0.3
    
    write_video_every: int = 10_000
    visualize_every: int = 25_000
    
    # Evaluation
    eval_every: int = 50_000
    eval_episodes: int = 4
    eval_horizon: int = 32
    eval_batch_size: int = 4
    max_eval_examples_to_plot: int = 4


@dataclass
class EvalBCRewExperimentConfig:
    bc_rew_ckpt_path: str = "logs/bc_rew"
    tokenizer_ckpt_path: str = "logs/tokenizer"
    
    # Head settings
    action_dim: int = 4
    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0
    reward_log_high: float = 3.0
    n_tasks: int = 128
    use_task_ids: bool = True
    
    # Sampler/eval
    ctx_length: int = 32
    horizon: int = 16
    schedule: str = "finest"  # "finest" or "shortcut"
    d: float | None = None
    ctx_signal_tau: float = 1.0
    match_ctx_tau: bool = False
    
    # Visualization
    max_examples_to_plot: int = 4
    paranoid_no_leak: bool = True

# --- Root Configs (Configuring the whole run) ---

@dataclass
class TokenizerTrainConfig:
    run_name: str = "tokenizer"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: TokenizerModelConfig = field(default_factory=TokenizerModelConfig)
    experiment: TokenizerExperimentConfig = field(default_factory=TokenizerExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

@dataclass
class DynamicsTrainConfig:
    run_name: str = "dynamics"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)
    experiment: DynamicsExperimentConfig = field(default_factory=DynamicsExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class BCRewTrainConfig:
    run_name: str = "bc_rew"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: BCRewModelConfig = field(default_factory=BCRewModelConfig)
    experiment: BCRewExperimentConfig = field(default_factory=BCRewExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class PolicyTrainConfig:
    run_name: str = "policy"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: PolicyModelConfig = field(default_factory=PolicyModelConfig)
    experiment: PolicyExperimentConfig = field(default_factory=PolicyExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class EvalBCRewConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: BCRewModelConfig = field(default_factory=BCRewModelConfig)
    experiment: EvalBCRewExperimentConfig = field(default_factory=EvalBCRewExperimentConfig)
