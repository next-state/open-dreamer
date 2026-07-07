# Dreamer 4 JAX World Model

**What is this?** An unofficial JAX/Flax NNX implementation of the Dreamer 4 world-model stack from [Training Agents Inside of Scalable World Models](https://danijar.com/project/dreamer4/). This checkout currently covers the world model: causal video tokenizer, offline dataset tokenization, action-conditioned latent dynamics training, and FVD evaluation. It does not yet include the full Dreamer 4 agent/RL training pipeline.

**Why should I care?** If you want to train or modify scalable action-conditioned video world models in JAX, this repo gives you the core pieces without a large framework around them: Grain + ArrayRecord data loading, Hydra configs, Orbax checkpoints, mixed precision, JAX sharding, and readable training scripts.

**How do I install it?**

```bash
pip install uv
uv sync
source .venv/bin/activate
```

The project targets Python 3.11 and CUDA 12 JAX. 

**How do I use it?** Create fixed-length MP4 ArrayRecord shards, train the tokenizer, tokenize the dataset into latent ArrayRecords, copy the latent mean/std into the latent dataset config, then train the dynamics model.

```bash
uv run python scripts/train_tokenizer.py dataset.array_record_path=/path/to/mp4-arrayrecords
uv run python scripts/tokenize_minecraft_dataset.py tokenizer_ckpt=logs/tokenizer-big/checkpoints dataset.array_record_path=/path/to/mp4-arrayrecords output_dir=/path/to/latents
uv run python scripts/train_dynamics.py tokenizer_ckpt=logs/tokenizer-big/checkpoints dataset.array_record_path=/path/to/latents dataset.dataloader_cfg.short_T=64 dataset.dataloader_cfg.long_T=256
```

**How can I contribute?** Keep changes focused on the world-model pipeline, update the relevant config/docs when behavior changes, and include the commands or logs you used to verify the change. Useful areas are dataset converters, correctness fixes, FVD/evaluation, sharding/performance work, reproducible training recipes, and documentation.

## Current Scope

Implemented:

- Causal video tokenizer trained as a masked or full-frame autoencoder.
- Minecraft/VPT-style MP4 ArrayRecord loading with keyboard/mouse action parsing.
- Tokenization script that writes latent ArrayRecord shards and latent statistics.
- Interactive dynamics model trained over tokenizer latents and shifted actions.
- Sampling/FVD evaluation for trained dynamics checkpoints.

Not implemented in this checkout:

- Agent tokens, behavior cloning heads, reward/value heads, and RL in imagination.
- A complete raw-video-to-MP4-ArrayRecord conversion script for your own dataset.

## Repository Layout

```text
.
├── dreamer/                 # Core models, training helpers, data, sampling, FVD
│   ├── data/                # Grain/ArrayRecord pipelines and serialization
│   ├── fvd/                 # FVD feature extraction and scoring
│   ├── models.py            # Tokenizer and dynamics model definitions
│   ├── training.py          # Training and evaluation helpers
│   ├── generation.py        # Denoising schedules and rollout utilities
│   └── checkpointing.py     # Orbax checkpoint bundles
├── scripts/
│   ├── train_tokenizer.py
│   ├── tokenize_minecraft_dataset.py
│   ├── train_dynamics.py
│   └── eval_fvd.py
├── configs/
│   ├── dataset/             # Raw-video and latent dataset configs
│   ├── tokenizer.yaml
│   ├── tokenize.yaml
│   ├── dynamics.yaml
│   └── eval_fvd.yaml
├── docs/                    # Paper notes, figures, and development notes
├── frontend/                # Experimental interactive frontend
└── reactor_app/             # Experimental Reactor sidecar
```

## Install

Install [`uv`](https://docs.astral.sh/uv/) first, then from the repository root:

```bash
pip install uv
uv sync
```

The dependency lock installs `jax[cuda12]`. If your machine needs a different JAX build, install the correct wheel for your CUDA/accelerator setup after `uv sync`.

## Dataset

The main workflow assumes Minecraft/VPT-style raw records:

- Files are named `shard-*.array_record`.
- Each ArrayRecord entry is a pickled Python dict.
- Each entry contains a fixed-length MP4 episode. Fixed length matters because tokenization batches full episodes.

Expected raw record shape:

```python
{
    "video": mp4_bytes,                  # bytes for one MP4 clip
    "video_shape": (T, H, W, C),          # used for length filtering
    "actions": [action_dict_0, ...],      # VPT-style action dicts, length T
    "source": "optional-id-or-path",
}
```

For a simple first run, make every record exactly 256 frames. Then use dynamics settings such as `short_T=64` and `long_T=256`.

Frame and shape constraints:

- `long_T` must be no larger than the number of frames in each record.
- Dynamics requires `long_T % short_T == 0`.
- Evaluation requires `ctx_length + horizon` frames and, when chunking FVD, `horizon % fvd_chunk_size == 0`.
- `H + padding_H[0] + padding_H[1]` and `W + padding_W[0] + padding_W[1]` must be divisible by `patch_size`.
- Global batch size `B` must be divisible by `jax.process_count()`.

Raw video settings live in [configs/dataset/minecraft_vpt.yaml](configs/dataset/minecraft_vpt.yaml). Update at least:

```yaml
array_record_path: /path/to/mp4-arrayrecords
index_max: 1500
H: 360
W: 640
padding_H: [4, 4]
padding_W: [0, 0]
patch_size: 16
dataset_mean: [0.2241, 0.2348, 0.2086]
dataset_std: [0.1809, 0.1874, 0.2282]
```

`dataset_mean` and `dataset_std` are pixel statistics for normalized video values. If you change the raw dataset, recompute them and put them in the raw dataset config before tokenizer training.

## Train The Tokenizer

The tokenizer learns the latent representation used by the dynamics model. The default config is [configs/tokenizer.yaml](configs/tokenizer.yaml).

```bash
ulimit -n 50000
uv run python scripts/train_tokenizer.py \
  run_name=tokenizer-big \
  dataset.array_record_path=/path/to/mp4-arrayrecords \
  dataset.index_max=1500 \
  dataset.dataloader_cfg.B=8 \
  dataset.dataloader_cfg.short_T=16 \
  dataset.dataloader_cfg.long_T=16
```

Outputs go to `logs/<run_name>/` by default:

- `checkpoints/` contains tokenizer and optimizer checkpoints.
- `vis/` contains reconstruction images when `visualize_every > 0`.

Tokenizer training can use shorter windows than the full raw episode length. For example, fixed 256-frame records can still train with 16-frame windows, then be tokenized as full 256-frame latent episodes.

## Tokenize The Dataset

After tokenizer training, encode each MP4 episode into tokenizer latents:

```bash
uv run python scripts/tokenize_minecraft_dataset.py \
  tokenizer_ckpt=logs/tokenizer-big/checkpoints \
  dataset.array_record_path=/path/to/mp4-arrayrecords \
  dataset.index_max=1500 \
  dataset.dataloader_cfg.B=8 \
  output_dir=/path/to/tokenized_data \
  records_per_shard=5000
```

This writes:

```text
/path/to/tokenized_data/
  shard-00000.array_record
  shard-00001.array_record
  metadata/latent_stats.npz
```

Print the latent statistics:

```bash
python - <<'PY'
import numpy as np
stats = np.load("/path/to/tokenized_data/metadata/latent_stats.npz")
print("latent_mean:", stats["mean"].tolist())
print("latent_std:", stats["std"].tolist())
print("num_samples:", int(stats["num_samples"]))
print("num_videos:", int(stats["num_videos"]))
PY
```

Copy `latent_mean` and `latent_std` into [configs/dataset/minecraft_vpt_latent.yaml](configs/dataset/minecraft_vpt_latent.yaml), and point `array_record_path` at the tokenized output directory.

## Train The Dynamics Model

The dynamics model trains on latent ArrayRecords and shifted actions. The default config is [configs/dynamics.yaml](configs/dynamics.yaml), which imports [configs/dataset/minecraft_vpt_latent.yaml](configs/dataset/minecraft_vpt_latent.yaml).

```bash
uv run python scripts/train_dynamics.py \
  run_name=dynamics-pre-final \
  tokenizer_ckpt=logs/tokenizer-big/checkpoints \
  dataset.array_record_path=/path/to/tokenized_data \
  dataset.index_max=89 \
  dataset.dataloader_cfg.B=8 \
  dataset.dataloader_cfg.short_T=64 \
  dataset.dataloader_cfg.long_T=256 \
  dataset.dataloader_cfg.long_ratio=0.1
```

For 256-frame tokenized records, `short_T=64` and `long_T=256` satisfy the packing constraint. If your records are shorter, reduce `long_T`; if you change `long_T`, keep `short_T` as a divisor.

## Evaluate

Generate videos and compute FVD from a dynamics checkpoint:

```bash
uv run python scripts/eval_fvd.py \
  dynamics_ckpt=logs/dynamics-pre-final/checkpoints \
  dataset.array_record_path=/path/to/mp4-arrayrecords \
  num_videos=256 \
  ctx_length=4 \
  horizon=240 \
  fvd_chunk_size=16
```

Use `mode=generate` to only save MP4s and `mode=evaluate` to compute FVD from previously generated videos.

## Configuration

Configs are Hydra/OmegaConf YAML files under `configs/`. You can edit YAML files or override values from the command line:

```bash
uv run python scripts/train_dynamics.py \
  max_steps=200000 \
  parallel_strategy=fsdp \
  logger.use_wandb=true \
  logger.wandb_project=dreamer4-jax
```

Useful config files:

- [configs/tokenizer.yaml](configs/tokenizer.yaml) - tokenizer architecture and training.
- [configs/tokenize.yaml](configs/tokenize.yaml) - offline tokenization.
- [configs/dynamics.yaml](configs/dynamics.yaml) - dynamics architecture and training.
- [configs/eval_fvd.yaml](configs/eval_fvd.yaml) - rollout and FVD settings.
- [configs/dataset/minecraft_vpt.yaml](configs/dataset/minecraft_vpt.yaml) - raw MP4 dataset settings.
- [configs/dataset/minecraft_vpt_latent.yaml](configs/dataset/minecraft_vpt_latent.yaml) - tokenized latent dataset settings.

## Contributing

Before opening a PR:

1. Make sure the README and relevant config comments still match the code.
2. Run the smallest command that exercises your change.
3. Include the command, hardware assumptions, and any important metrics in the PR.
4. Do not add new framework layers unless they remove real complexity from the training path.

High-value contributions:

- Raw dataset converters that emit the MP4 ArrayRecord schema above.
- Small regression tests or smoke tests for data loading, tokenization, and checkpoint restore.
- Multi-host and sharding fixes.
- Evaluation scripts and reproducible config recipes.
- Documentation for known-good runs.

## References

- Dreamer 4: [Training Agents Inside of Scalable World Models](https://danijar.com/project/dreamer4/)
- Jasmine: [A simple, performant and scalable JAX-based world modeling codebase](https://github.com/p-doom/jasmine)
