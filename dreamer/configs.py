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
    run_name: str = "default"
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

# --- Root Configs (Configuring the whole run) ---

@dataclass
class TokenizerTrainConfig:
    # Components
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: TokenizerModelConfig = field(default_factory=TokenizerModelConfig)
    experiment: TokenizerExperimentConfig = field(default_factory=TokenizerExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

@dataclass
class DynamicsTrainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: DynamicsModelConfig = field(default_factory=DynamicsModelConfig)
    experiment: DynamicsExperimentConfig = field(default_factory=DynamicsExperimentConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
