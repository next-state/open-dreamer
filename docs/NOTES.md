# Dreamer 4 JAX — Codebase Notes

This is a JAX/Flax NNX reproduction of **Dreamer 4** (Hafner, Yan, Lillicrap),
a scalable agent that solves control tasks by RL inside a fast world model.
The paper trains a 2B-param system (400M tokenizer + 1.6B dynamics) on Minecraft
VPT data and is the first agent to obtain diamonds purely from offline data.

---

## Architecture Overview (from the paper)

The system has three stages:

1. **Causal Tokenizer** — Encoder/decoder transformer with a continuous
   bottleneck. Trained with masked autoencoding (MAE, random patch dropout
   0–90%) plus MSE + LPIPS reconstruction loss.

2. **Interactive Dynamics** — Operates on interleaved latents and actions.
   Trained with *shortcut forcing* (diffusion forcing + shortcut models) using
   x-prediction and a ramp loss weight. Generates frames in K=4 denoising steps
   for real-time inference on one GPU.

3. **Imagination Training** — Task-conditioned policy and reward heads are
   finetuned into the frozen dynamics transformer via agent tokens. Policy is
   then improved with PMPO reinforcement learning on imagined rollouts.

The transformer is a 2D block-causal design (space + time axes) with RoPE,
SwiGLU, QKNorm, GQA, sparse temporal layers (every 4th layer), and register
tokens.

---

## Repository Structure

```
dreamer4-jax-private/
├── dreamer/                # Core library
│   ├── models.py           # All neural network modules (~1500 lines)
│   ├── training.py         # Loss functions (shortcut forcing, BC, PMPO, TD-lambda)
│   ├── generation.py       # Autoregressive rollout with KV cache
│   ├── configs.py          # Hydra-compatible dataclasses
│   ├── actions.py          # VPT action space (23 binary keys + 121 camera classes)
│   ├── data/               # Grain dataloaders, transforms, serialization
│   ├── checkpointing.py    # Orbax checkpoint bundles (tokenizer/dynamics/heads)
│   ├── parallel.py         # Mesh strategies (data, FSDP, TP, sequence parallel)
│   ├── sampler.py          # Visualization sampling wrapper
│   ├── scaling.py          # Iso-FLOPs / tokens-per-param scaling utilities
│   ├── logging.py          # W&B logger
│   └── utils.py            # TokenLayout, RunningNormalizer, LR schedules, optimizers
├── scripts/                # Training entry points
│   ├── train_tokenizer.py          # Stage 1 — tokenizer pretraining
│   ├── train_dynamics.py           # Stage 2 — dynamics pretraining
│   ├── train_heads.py              # Stage 3 — BC + reward finetuning
│   ├── train_policy.py             # Stage 4 — RL (BROKEN, legacy imports)
│   ├── new_train_policy.py         # Stage 4 — RL (BROKEN, missing modules)
│   ├── tokenize_minecraft_dataset.py  # Offline latent encoding pipeline
│   ├── eval_dynamics.py            # Standalone evaluation
│   └── check_diffusion_math.py     # Math verification test
├── configs/                # Hydra YAML configs
│   ├── tokenizer.yaml / dynamics.yaml / heads.yaml / policy.yaml
│   └── dataset/            # coinrun, minecraft_vpt, minecraft_vpt_latent, 160x90 variants
├── reactor/                # Interactive frontend (CoinRun demo)
└── frontend/               # Web UI (pnpm)
```

---

## What's Working

| Component | Script | Status |
|-----------|--------|--------|
| Tokenizer pretraining (MAE + LPIPS) | `train_tokenizer.py` | Working |
| Dynamics pretraining (flow loss) | `train_dynamics.py` | Working |
| Offline tokenization (Minecraft VPT) | `tokenize_minecraft_dataset.py` | Working |
| Agent finetuning (BC + reward + dynamics) | `train_heads.py` | Working |
| Evaluation & video generation | `eval_dynamics.py` | Working |
| KV-cache autoregressive rollout | `generation.py` | Working |
| All model architectures | `models.py` | Working |
| Data loading (CoinRun, Minecraft, latent) | `data/` | Working |
| Checkpointing & resume | `checkpointing.py` | Working |
| Parallelism (data/FSDP/TP/SP) | `parallel.py` | Working |
| W&B logging + scaling analysis | `logging.py`, `scaling.py` | Working |

---

## What's Broken / Incomplete

| Issue | Location | Details |
|-------|----------|---------|
| RL training (PMPO) | `train_policy.py`, `new_train_policy.py` | Both import non-existent modules (`ParallelContext`, `ImaginationConfig`, `make_manager`, `MetricLogger`). No working RL loop. |
| Bootstrap loss unused in dynamics | `train_dynamics.py` | Always sets `B_self=0`, so only flow loss runs. Bootstrap only activates in `train_heads.py`. |
| Ramp loss weight unused | `training.py` | `ramp_weight()` function exists but is never called in the loss computation. |
| Actions dropped from VPT video loader | `data/transforms.py` | `ProcessMinecraftEpisodeAndSlice` has a `# FIXME: no actions returned!!` — returns empty `Actions`. Latent data path works fine. |
| Multi-task support | `train_heads.py` | Hardcodes `task_id=0` for all samples. Paper describes 20 Minecraft tasks. |
| Attention logit soft capping | `models.py` | Paper mentions it; not implemented. |

---

## Key Model Details (models.py)

### Tokenizer
- **Encoder**: patchify → linear project → MAE mask replacement → prepend N learned latent
  tokens → BlockCausalTransformer (encoder space mask) → bottleneck linear + tanh
- **Decoder**: up-project bottleneck → concat with learned patch queries →
  BlockCausalTransformer (decoder space mask) → patch head → unpatchify
- Shared `pixel_normalizer` (EMA RunningNormalizer) and `latent_normalizer`

### Dynamics
- Token layout per timestep: `[action(1), shortcut(1), spatial(n_spatial), register(n_reg), agent(n_agent)]`
- Latents are packed by `packing_factor` (default 2) to reduce sequence length
- Shortcut conditioning: discrete embeddings for step size d and signal level tau,
  concatenated into a single token
- Output: zero-initialized `flow_x_head` projects spatial states back to latent space
- Space mask (`"wm_agent"`): actions attend to actions only, obs attend to obs+actions,
  agent tokens attend to everything — prevents causal confusion

### Heads
- **PolicyHeadMTP**: multi-token prediction (L=8 future actions). SwiGLU MLP from agent
  hidden states → separate binary/categorical/continuous output heads
- **RewardHeadMTP**: same structure, outputs symexp twohot categorical bins
- **ValueHead**: single-step symexp twohot output

### Shortcut Forcing (training.py)
- **Flow loss** (at finest step d_min=1/k_max): corrupt latents z_tilde = (1-sigma)*z0 + sigma*z1,
  predict clean z1, MSE loss
- **Bootstrap loss** (at coarser steps): run full-step prediction vs two half-steps,
  MSE in v-space scaled by (1-sigma)^2
- Both losses combined in `shortcut_forcing_step()` with configurable `B_self` split

---

## Config Defaults

| Parameter | Tokenizer | Dynamics |
|-----------|-----------|----------|
| d_model | 1024 (enc) / 512 (dec) | 1024 |
| depth | 8 (enc) / 4 (dec) | 16 |
| n_heads | 16 (enc) / 8 (dec) | 16 |
| n_kv_heads | same | 4 (GQA) |
| time_every | 4 | 4 |
| n_latents | 512 | — |
| d_bottleneck | 16 | — |
| packing_factor | — | 2 (512 latents → 256 spatial tokens) |
| n_register | — | 8 |
| k_max | — | 64 |
| context_length | — | 192 frames |
| optimizer | Muon, lr=3e-3 WSD | Muon, lr=1e-2 WSD |
| dtype | bfloat16 | bfloat16 |

---

## Training Pipeline

```
Stage 1: train_tokenizer.py    →  TokenizerCheckpointBundle
                                      ↓
Stage 2: tokenize_minecraft_dataset.py  →  latent shards (.array_record)
                                      ↓
Stage 3: train_dynamics.py     →  DynamicsCheckpointBundle (includes frozen tokenizer)
                                      ↓
Stage 4: train_heads.py        →  HeadsCheckpointBundle (BC + reward heads)
                                      ↓
Stage 5: train_policy.py       →  RL with PMPO  [NOT YET WORKING]
```

---

## Datasets

- **CoinRun**: 64x64, 16 categorical actions, pickle format. Simple test environment.
- **Minecraft VPT**: 360x640 (or 160x90 downsampled), 23 binary + 121 categorical actions,
  MP4 shards. 2541 hours of contractor gameplay at 20 FPS.
- **Latent (pre-tokenized)**: msgpack-serialized latents + actions in ArrayRecord shards.
  Generated by `tokenize_minecraft_dataset.py`. Avoids re-encoding during dynamics training.
