# Pixel Dynamics Stability Experiments (Feb 20, 2026)

## Goal
Fix generation degradation in pixel-space dynamics rollouts (artifact buildup/saturation/smearing over horizon).

## Baseline
- Reference run: `wandb.ai/pal/tiny_dreamer_4/runs/jn7e9we4`
- Local run dir: `logs/dynamics-high-lr`
- Current known artifact: right-column generations in `step_023000/diffusion_grid.mp4` degrade over time (speckle noise, blur, washed textures).

## Hypotheses
1. Missing bootstrap consistency in dynamics pretraining (`B_self=0`) causes denoising updates to be inconsistent across step decomposition, leading to autoregressive error amplification.
2. Denoising dynamics are not contractive enough in normalized pixel space; per-step residuals can drive moment drift (mean/std), perceived as saturation/exposure blow-up.
3. Inference update can be stabilized with mathematically controlled blending/normalization, reducing drift without retraining.

## Plan
1. Baseline characterization from latest checkpoint (qualitative + simple drift statistics).
2. Inference-only ablations (using existing checkpoint): conservative update scaling / clamps.
3. Training-time fixes: enable bootstrap rows + add stability regularization.
4. Resume training from baseline checkpoint in tmux; compare eval videos and metrics every 1k steps.

## Experiment Log
- [00:00] Created branch `stability-fix-pixel-generation` and initialized this log.
- [00:01] Baseline drift quantification on `logs/dynamics-high-lr/viz/step_023000/diffusion_grid.mp4`:
  - Right-column (generated) mean intensity drift: `36.00 -> 63.92` (+27.92) vs left-column reference `35.62 -> 52.66` (+17.04).
  - Right/left mean ratio grows from `1.01` to `1.21` by final frame.
  - Right-column dark-pixel fraction (`<=5`) increases from `0.0419` to `0.0854`; bright-pixel fraction (`>=250`) from `0.0023` to `0.0080`.
  - Strong artifact growth in rows 1-2 (grass/dungeon), consistent with visual speckle/saturation blow-up.
- [00:02] Implemented inference-time denoising stabilizer (optional):
  - Added schedule knobs: `denoise_update_scale`, `denoise_max_residual_rms`, `denoise_state_clip`.
  - Update now supports trust-region residual scaling: `x_{t+1} = x_t + eta * mix * clipped_residual`.
- [00:03] Built checkpoint evaluator: `scripts/eval_pixel_dynamics_ckpt.py` for fast ablations from existing checkpoints.
- [00:04] Inference ablations from checkpoint step 25k (same batch/seed):
  - Baseline: shortcut MSE `0.0526`, diffusion MSE `0.0323`.
  - `update_scale=0.90`: shortcut MSE `0.0500`, diffusion MSE `0.0215` (best).
  - `update_scale=0.90, max_residual_rms=1.2`: shortcut MSE `0.0501`, diffusion MSE `0.0221`.
  - `update_scale=0.85, max_residual_rms=1.0, clip=4.0`: shortcut MSE `0.0493`, diffusion MSE `0.0271`.
  - `update_scale=0.95, max_residual_rms=1.5`: shortcut MSE `0.0512`, diffusion MSE `0.0262`.
- [00:05] Conclusion so far: conservative blend (`update_scale=0.90`) materially improves diffusion rollout quality from the same weights.
- [00:06] Implemented training fix candidates:
  - Enabled configurable bootstrap rows in `train_dynamics.py` via `bootstrap_frac` and `bootstrap_start`.
  - Added configs:
    - `configs/dynamics_stability_bootstrap.yaml`
    - `configs/dynamics_stability_bootstrap_low_lr.yaml`
- [00:07] Long-horizon (T=64, horizon=60) checkpoint ablation:
  - Baseline diffusion MSE `0.0657`.
  - `update_scale=0.90` diffusion MSE `0.0312` (large reduction, ~52%).
- [00:08] Bootstrap training integration test:
  - `bootstrap_frac=0.5` on single GPU OOM (expected due 3x forward path for bootstrap branch).
  - `bootstrap_frac=0.25` on 4 GPUs successful; short smoke run completed and checkpoint `26109` saved.
- [00:09] Launched long run in tmux:
  - Session: `dyn_stab_v1`
  - Command: `train_dynamics.py --config-name dynamics_stability_bootstrap run_name=dynamics-stability-bootstrap-v1`
  - GPUs: `4,5,6,7`
  - Resume checkpoint: step `27000`
  - W&B run: `https://wandb.ai/pal/tiny_dreamer_4/runs/av1ukyhy`
