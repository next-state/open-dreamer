# The problem

right now the rollout trajectory is not stable.
what happens is that over time the generation quality gets worse.
it's almost as if the attractor is a jumbled mess


# Strategies tried
## Sweep over tau_ctx
- Added evaluation sweeps over `tau_ctx_target`, including a coarse `0.10-0.99` sweep and a finer `0.50-0.70` sweep, with per-tau rollout videos for online and EMA comparison.
- Result: Failed. no value of tau_ctx solves the problem
- This suggests that the problem is not the guidance. (the noiser the context, the less used it is.)

## Pixel collapse.
- Tried if by denoising -> decoding -> encoding -> final latent the generation quality improves.
- Result: Failed. the rollouts are even worse and much slower as well
- This suggests that the problem is not clippig (by decoding and re-encoding you effectively clip the latent space)
- This suggests that the problem is not solved by grounding the latents with the tokenizer (such as in llms for reasoning models)

## 