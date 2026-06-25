# Dreamer 4 World Models, in pure JAX

This repo is an unofficial implementation of the **[Dreamer 4](https://danijar.com/project/dreamer4/)** world model and RL agent from *“Training Agents Inside of Scalable World Models”* in pure JAX. This repo is designed to be educational and serve as a starting point for those interested in world models, RL, and Jax. 

![dreamer4](docs/architecture.png)

At this stage, the entire world model + RL pipeline has been implemented and tested on recorded trajectory datasets like CoinRun. The authors are extending the codebase to solve harder tasks and eventually, Minecraft.  

> [!NOTE]
> We are looking for support - in terms of compute, advising, or feature development. Please get in touch if interested!


- [Website](https://danijar.com/project/dreamer4/)
- [Twitter](https://x.com/danijarh/status/1973072288351396320)

## Demo

At a high level, Dreamer 4 first trains an action-conditioned video diffusion model of the environment. Here, we show that the world model has learned to accurately predict environment dynamics from recorded trajectories.
<figure>
    <img src="docs/imagination-cropped.gif">
</figure>


Then, the agent is trained with RL in the world model. The reward is the proximity to the center of the image. We can see that the agent successfully learns to hover near the center.
<figure>
    <img src="docs/rl-cropped.gif">
</figure>



## Repo Structure
- **Core library (`dreamer/`)**
  - `models.py` -- Space-time axial attention, causal tokenizer, interactive dynamics model, agent / reward / value heads.
  - `data.py` -- Recorded trajectory data loading (ArrayRecord format) 
  - `imagination.py` -- JIT-fused imagination / diffusion-style rollout code used for fast RL in latent space.
  - `sampler.py` - non-JIT sampling helpers for debugging / visualization.
  - `utils.py` -- training state helpers, checkpointing wrappers (Orbax), logging helpers.
- **Training & evaluation scripts (`scripts/`)**
  - `train_tokenizer.py` -- Trains the causal tokenizer (masked autoencoder over video).
  - `train_dynamics.py` -- Trains the interactive dynamics model on top of the frozen tokenizer.
  - `train_heads.py` -- Adds behavior cloning and reward prediction heads on the world model (agent tokens + reward head).
  - `train_policy.py` -- Runs Dreamer‑style RL purely in imagination using the learned world model and heads (PMPO-style update).
- **Docs & logs**
  - `docs/` -- Figures, videos, and notes from development (e.g., reconstructions, imagination rollouts).

This repo is intentionally built to be easy to modify. If you want to understand or change the algorithm, start from the training scripts under `scripts/` and follow the calls into `dreamer/`.

## Setup

We use `uv` to manage the environment and dependencies (see `pyproject.toml` / `uv.lock`):

```bash
uv sync      # creates .venv and installs packages
source .venv/bin/activate # activate venv
uv pip install -e . # install this project as an editable package
```

The code should run on any relatively recent GPU, but all logic should also run (more slowly) on CPU.

## Training Pipeline
Dreamer 4 follows a 4-stage training pipeline.
- **Phase 1**: Train a causal tokenizer (MAE-style) on videos.
- **Phase 2**: Train an interactive dynamics model in latent space of the tokenizer.
- **Phase 3**: Add agent tokens, BC / reward heads with behavior cloning and reward prediction.
- **Phase 4**: Train a policy on imagination trajectories from the dynamics model.

The default experiments use recorded trajectory datasets in ArrayRecord format (e.g., CoinRun episodes).

To run the training pipeline, edit the configs in each script's `__main__` block and execute:

```bash
# Phase 1: Train the causal tokenizer
python scripts/train_tokenizer.py

# Phase 2: Train the dynamics model (requires tokenizer checkpoint)
python scripts/train_dynamics.py tokenizer_ckpt=./logs/tokenizer/checkpoints

# Phase 3: Train BC/reward heads (requires tokenizer + dynamics checkpoints)
python scripts/train_heads.py tokenizer_ckpt=./logs/tokenizer/checkpoints dynamics_ckpt=./logs/train_dynamics/checkpoints

# Phase 4: Train policy in imagination (requires BC/reward checkpoint)
python scripts/train_policy.py heads_ckpt=./logs/heads/checkpoints
```

All scripts save checkpoints under `logs/{run_name}/checkpoints/` by default. You can also enable wandb logging by adding `use_wandb=True` to the launch command. We use hydra to manage the configurations in `/configs`.

## Learned perturbation matching (Qφ)

An optional dynamics-training mode that *learns* the perturbation injected into context
frames instead of using fixed Gaussian forcing. A small causal network `Qφ` models the
world model's per-frame error distribution `p(e | z, t)`; we sample from it and add the
perturbation to the context, so the world model learns to denoise realistic,
content-dependent context errors. It targets autoregressive **exposure bias** (at inference
the model conditions on its own imperfect frames). See `dreamer/qphi.py`,
`QphiModelConfig`, and the integration in `shortcut_forcing_step` / `train_dynamics.py`.

Key properties: `Qφ` outputs an explicit per-frame distribution (low-rank-plus-diagonal
Gaussian base + identity-initialised normalizing flow) trained by exact `log_prob`; it has a
**separate optimizer**; the injected perturbation is **detached** into the world model (so
`L_world` cannot collapse `Qφ`) and the matching target `e` is **stop-gradient**. It is
gated behind `qphi.enabled` — with it `false` (the default) training is bit-for-bit the
vanilla baseline.

`Qφ` is injected from step 0 with **no warmup schedule**: it is initialised to ~zero
perturbation (small `qphi.s_init`) and the variance grows from below via the matching loss,
so the perturbation enters small and increases only as the learned error warrants. When a
perturbation-matched model is rolled out, the generated frame is used **as-is** (no
diffusion-forcing re-noise): `run_evaluation` and `eval_exposure_bias.py` set
`tau_ctx_target=1.0` (clean context) automatically/with a flag.

**Two-stream attention.** The world model runs two aligned streams: a **query/target**
stream carrying the diffusion-noised input being denoised, and a **context** stream carrying
`z + λ·pert` (clean latent + the Qφ perturbation), conditioned as clean. In time-attention
the query **cross-attends strictly causally** to the context (positions `< t`, never its own
diagonal — so it can't trivially copy the clean target), while the context stream
self-attends to build its representation. The prediction comes from the query stream. This
cleanly separates the two roles (the context is clean + Qφ, the target stays σ-noised) at
~2× dynamics compute; it's gated on `qphi.enabled`, and single-stream (baseline, tokenizer,
inference) is unchanged. At rollout, set `tau_ctx_target=1.0` so the generated context is
clean (matching training).

> Note on this repo's σ convention: `σ` is a *signal* level (`σ=1` clean, `σ=0` max noise),
> inverted vs. the usual diffusion `t`. `qphi.t_query` selects the operating point at which
> `Qφ` is sampled for injection (in signal-σ).

```bash
# Baseline (vanilla diffusion forcing) — identical to leaving qphi out entirely
python scripts/train_dynamics.py tokenizer_ckpt=./logs/tokenizer/checkpoints qphi.enabled=false

# Ablations (paper config points). type ∈ {none, gaussian_iso, gaussian_lowrank, flow}:
python scripts/train_dynamics.py tokenizer_ckpt=... qphi.enabled=true qphi.type=none              # fixed-Gaussian forcing
python scripts/train_dynamics.py tokenizer_ckpt=... qphi.enabled=true qphi.type=gaussian_iso      # learned isotropic
python scripts/train_dynamics.py tokenizer_ckpt=... qphi.enabled=true qphi.type=gaussian_lowrank  # learned anisotropic
python scripts/train_dynamics.py tokenizer_ckpt=... qphi.enabled=true qphi.type=flow              # full module
# lambda sweep on any of the above (over-provision robustness; prior λ ≥ 1):
python scripts/train_dynamics.py tokenizer_ckpt=... qphi.enabled=true qphi.type=flow qphi.lam=1.5
```

Training logs `qphi/pert_norm`, `qphi/e_norm` (matching-sanity / anti-collapse monitors),
and `qphi/loss`, `qphi/grad_norm`. Expect `pert_norm` to start near zero and grow toward
`e_norm` as `Qφ` learns.

**Exposure-bias rollout eval (the payoff metric).** Tune `λ` on this rollout metric, never
on `qphi/loss` (which is teacher-forced and cannot see the rollout gap). Run it per
checkpoint and overlay the CSVs:

```bash
python scripts/eval_exposure_bias.py dynamics_ckpt=./logs/<run>/checkpoints \
    ctx_length=8 horizon=32 num_steps=4 tag=<none|lowrank|flow|lam1.5> \
    output_dir=eval_outputs/exposure_bias
```

It writes per-frame normalised-latent-MSE and decoded-PSNR curves (`*.csv` + `*.png`).
Success = lower error growth over the horizon than `type=none` (vanilla DF).

Acceptance tests for all of the above (baseline reproducibility, gradient isolation, detach,
anti-collapse, exact density) live in `tests/test_qphi.py` (`pytest tests/test_qphi.py`).

## Experimental Results
A log of the training process.

#### MAE training
The MAE, after training, should have around 40 PSNR, and the visualizations should show perfect reconstruction from masked inputs.


<figure>
    <img src="docs/step_75900.png">
  <figcaption>Ground Truth, Masked Input, Reconstructions</figcaption>
</figure>

#### Dynamics training
The dynamics model is trained. It should get around ~30 PSNR in the autoregressive generations using diffusion, and ~29 PSNR using shortcut. The generations should look almost pixel perfect.

<figure>
    <img src="docs/dynamics_training.png">
  <figcaption>Dynamics model training curves. </figcaption>
</figure>

#### Reward / BC training
Agent tokens, Reward / BC heads are trained on top of the dynamics model, and the dynamics model is finetuned to prevent collapse.
<figure>
    <img src="docs/bc_rew_training.png">
  <figcaption>Reward / BC / Dynamics model losses. </figcaption>
</figure>

#### RL in imagination
The policy is trained with RL on imagination rollouts from the dynamics model.
<figure>
    <img src="docs/rl_training.png">
  <figcaption>RL training curves. The returns increase and the policy stays in the center. </figcaption>
</figure>



## References and Acknowledgements
This implementation references:
- **Dreamer 4**: [“Training Agents Inside of Scalable World Models”](https://danijar.com/project/dreamer4/)
- **Jasmine** ["Jasmine: A simple, performant and scalable JAX-based world modeling codebase"](https://github.com/p-doom/jasmine)

The authors would like to thank Danijar and the Jasmine team (Mihir, Franz, Alfred) for their advice. 