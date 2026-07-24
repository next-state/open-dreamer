<div align="center">

<img src="assets/banner.png" alt="Open Dreamer" width="100%" />

**An open, real-time implementation of the Dreamer 4 world-model pipeline in JAX/Flax.**

[🌐 Website & Blog Post](https://next-state.github.io/open-dreamer/) &nbsp;·&nbsp;
[🎮 Live Demo](https://next-state.github.io/open-dreamer/) &nbsp;·&nbsp;
[⚡ Inference Code](https://github.com/reactor-team/open-dreamer)

</div>

---

Open Dreamer is a simple, performant, and easy-to-use JAX/Flax NNX
implementation of the Dreamer 4 world model — trained on Minecraft/VPT-style
gameplay and playable in real time. This repository holds the **training
pipeline**: a causal video tokenizer, an action-conditioned latent dynamics
model, and the tools to generate rollouts and measure quality.

This repo currently supports:

- Training a causal video tokenizer
- Tokenizing Minecraft/VPT-style MP4 datasets
- Training an action-conditioned latent dynamics model
- Generating rollouts and computing FVD

## 🎮 Try it now

The fastest way to experience Open Dreamer is the **live, in-browser demo** —
step into a generated Minecraft world and play it in real time, with a
Game ⟷ Dream toggle that hands the stream from the real game to the world
model frame by frame. No setup required; the real-time model runs in the cloud
on [Reactor](https://reactor.inc).

👉 **[Open the demo](https://next-state.github.io/open-dreamer/)**

The [website](https://next-state.github.io/open-dreamer/) also walks through how
the model works, with interactive figures and result galleries.

## ⚡ Run it yourself

To roll out the trained model locally on your own video and actions — generating
new frames from an MP4 and a matching action sequence — use the inference
repository:

👉 **[reactor-team/open-dreamer](https://github.com/reactor-team/open-dreamer)**

The rest of this README covers **training** the model from scratch.

## 📋 Roadmap

- [ ] Full Dreamer 4 Behaviour-Cloning / RL agent training loop

## ⚙️ Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- CUDA 12-compatible JAX environment
- Minecraft/VPT-style ArrayRecord data; see [dreamer/data/README.md](dreamer/data/README.md)

## Install

```bash
pip install uv
uv sync
source .venv/bin/activate
```

The dependency lock targets CUDA 12 JAX. If your machine needs a different JAX
build, install the correct wheel for your accelerator setup after syncing.

## Workflow

1. Prepare raw MP4 ArrayRecord shards.
2. Train the tokenizer on raw video clips.
3. Tokenize full episodes into latent ArrayRecords.
4. Copy the generated latent statistics into the latent dataset config.
5. Train the dynamics model on latent episodes and actions.
6. Generate videos and compute FVD.

The commands below assume fixed 256-frame raw records. The example values such
as `index_max`, `short_T`, `long_T`, `horizon`, and `fvd_chunk_size` are explained
in [dreamer/data/README.md](dreamer/data/README.md).

## Repository layout

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
└── site/                    # Website, blog post, and the live demo (Next.js)
```

## Dataset

The main workflow expects raw Minecraft/VPT-style shards named
`shard-*.array_record`. Each raw record is a pickled Python dict containing MP4
bytes, a video shape, actions, and optional source metadata. Tokenized records
are written as msgpack ArrayRecords with tokenizer latents and actions.

Raw video settings live in
[configs/dataset/minecraft_vpt.yaml](configs/dataset/minecraft_vpt.yaml). Update
at least:

```yaml
array_record_path: /path/to/mp4-arrayrecords
index_max: 1500
dataset_mean: [0.2241, 0.2348, 0.2086]
dataset_std: [0.1809, 0.1874, 0.2282]
```

`dataset_mean` and `dataset_std` are pixel statistics for normalized video
values. Recompute them when changing the raw dataset.

## Train the tokenizer

The tokenizer learns the latent representation used by the dynamics model. The
default config is [configs/tokenizer.yaml](configs/tokenizer.yaml). Edit that
file and [configs/dataset/minecraft_vpt.yaml](configs/dataset/minecraft_vpt.yaml)
before running:

```bash
uv run scripts/train_tokenizer.py
```

Outputs go to `logs/<run_name>/` by default:

- `checkpoints/` contains tokenizer and optimizer checkpoints.
- `vis/` contains reconstruction images when `visualize_every > 0`.

Tokenizer training can use shorter windows than the full raw episode length.
For example, fixed 256-frame records can train with 16-frame windows and later
be tokenized as full 256-frame latent episodes.

## Tokenize the dataset

After tokenizer training, encode each MP4 episode into tokenizer latents. Edit
[configs/tokenize.yaml](configs/tokenize.yaml) first:

```bash
uv run scripts/tokenize_minecraft_dataset.py
```

This writes latent shards and statistics:

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

Copy `latent_mean` and `latent_std` into
[configs/dataset/minecraft_vpt_latent.yaml](configs/dataset/minecraft_vpt_latent.yaml),
then point `array_record_path` and `index_max` at the tokenized output.

## Train the dynamics model

The dynamics model trains on latent ArrayRecords and shifted actions. The
default config is [configs/dynamics.yaml](configs/dynamics.yaml), which imports
[configs/dataset/minecraft_vpt_latent.yaml](configs/dataset/minecraft_vpt_latent.yaml).
Edit both files before running:

```bash
uv run scripts/train_dynamics.py
```

## Evaluate

Generate videos and compute FVD from a dynamics checkpoint. Edit
[configs/eval_fvd.yaml](configs/eval_fvd.yaml) first:

```bash
uv run scripts/eval_fvd.py
```

Use `mode=generate` to only save MP4s and `mode=evaluate` to compute FVD from
previously generated videos.

## Configuration

Configs are Hydra/OmegaConf YAML files under `configs/`. The main script
configs are commented with the purpose and constraints for each field.

Useful config files:

- [configs/tokenizer.yaml](configs/tokenizer.yaml) - tokenizer architecture and training.
- [configs/tokenize.yaml](configs/tokenize.yaml) - offline tokenization.
- [configs/dynamics.yaml](configs/dynamics.yaml) - dynamics architecture and training.
- [configs/eval_fvd.yaml](configs/eval_fvd.yaml) - rollout and FVD settings.
- [configs/dataset/minecraft_vpt.yaml](configs/dataset/minecraft_vpt.yaml) - raw MP4 dataset settings.
- [configs/dataset/minecraft_vpt_latent.yaml](configs/dataset/minecraft_vpt_latent.yaml) - tokenized latent dataset settings.

## 📚 References

- Dreamer 4: [Training Agents Inside of Scalable World Models](https://danijar.com/project/dreamer4/)
- Jasmine: [A simple, performant and scalable JAX-based world modeling codebase](https://github.com/p-doom/jasmine)

## 📄 Citation

If you use Open Dreamer in your research, please cite:

```bibtex
@misc{marti2026opendreamer,
    title={How to Train a Frontier-level World Model},
    author={Marti Monso, Diego and Sacco, Francesco and Hu, Edward},
    month={jul},
    year={2026},
    publisher={Zenodo},
    doi={10.5281/zenodo.21475232},
    url={https://next-state.github.io/open-dreamer/},
}
```

Please also cite the original Dreamer 4 paper:

```bibtex
@misc{hafner2025trainingagentsinsidescalable,
    title={Training Agents Inside of Scalable World Models}, 
    author={Danijar Hafner and Wilson Yan and Timothy Lillicrap},
    year={2025},
    eprint={2509.24527},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2509.24527}, 
}
```
