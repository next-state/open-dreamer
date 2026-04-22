# Loss Spikes in Deeper Dynamics Models: Diagnosis & Recommendations

**Background:** We observe loss spikes when training the dynamics model at 24 and 30 layers, while 16 layers trains stably. After a spike, video quality degrades even once the loss curve recovers. We investigated using the "Spike No More" paper (Takase et al.), which provides a theoretical framework for diagnosing gradient explosions in Pre-LN transformers.

---

## Root Cause 1: No Depth-Scaled Output Projections

The paper shows that in Pre-LN transformers, if the output of each sublayer has `std ≈ 1`, residual variance grows **exponentially with depth**:

> `var(residual after L layers) ≈ 2^L`

Our model hits this condition. Both the attention output projection (W_O, `to_out`) and the FFN second layer (W_2, `fc_out`) use `lecun_normal` initialization, which is variance-preserving by design — so each layer approximately doubles the residual variance.

The standard fix (used in Megatron-LM, BLOOM, etc.) is to initialize W_O and W_2 with `N(0, σ / sqrt(2N))` where N is the number of layers. We don't do this.

**Why 16 layers survives but 24/30 don't:** The explosion is exponential:

| Depth | Variance amplification | Practical effect |
|---|---|---|
| 16 | ~2^16 ≈ 65,000× | AGC mostly absorbs it |
| 24 | ~2^24 ≈ 16,000,000× | Gradients regularly overwhelm AGC |
| 30 | ~2^30 ≈ 1,000,000,000× | Near-certain explosions |

**Compounding factor:** Our config ties `d_model = 64 × depth`, so going from 16 to 30 layers simultaneously increases both depth and width. This means more layers AND larger matrices, without any compensating change to initialization.

---

## Root Cause 2 (Secondary): Heterogeneous Embedding Std

The transformer input concatenates tokens with very different initial standard deviations:

| Token type | Init std |
|---|---|
| Spatial tokens (video latents) | ~1.0 (via `lecun_normal` + `spatial_proj`) |
| Shortcut token (`step_embed`) | 1.0 |
| **Action token** (`base_action_emb`) | **0.02** |
| **Register tokens** | **0.02** |

The paper shows that when inputs to a LayerNorm/RMSNorm have `std << 1`, the gradient is amplified by O(1/std). Action and register tokens at std=0.02 cause the first RMSNorm to amplify gradients by ~50× for those positions. There is no embedding normalization step before the transformer stack to correct this.

---

## What's Already Helping (But Not Enough)

- **Zero-init on `flow_x_head`** — the model correctly starts near a no-op output
- **Pre-LN (RMSNorm before every sublayer)** — inherently more stable than Post-LN at depth
- **Adaptive Gradient Clipping (AGC, clip=0.3)** — provides per-parameter protection, but is treating the symptom rather than the cause; it gets overwhelmed at 24/30 layers

---

## Recommendations

**Fix 1 — Depth-scale W_O and W_2 (addresses Root Cause 1):**

In `dreamer/models.py`, change the kernel init for `to_out` (attention output projection) and `fc_out` (FFN second layer) to include a `1/sqrt(2*depth)` factor. This ensures each layer's contribution to the residual stream is O(1/N), bounding total variance regardless of depth. This is the "scaled initialization" from the paper and is standard in large LLM training.

**Fix 2 — Add Embed LN before the transformer stack (addresses Root Cause 2):**

Apply an `RMSNorm` to the full input sequence after token concatenation in `Dynamics.__call__`, before it enters `BlockCausalTransformer`. This normalizes all token types (including action tokens at std=0.02) to std=1 before the first attention layer. The paper calls this "Embed LN" and shows it is theoretically sufficient to prevent the LN gradient explosion.

An alternative to Fix 2 is "Scaled Embed": multiply `base_action_emb` and `register_tokens` by `sqrt(d_model)` at the point of use, bringing their effective std from 0.02 up to ~1. This is what the original "Attention is All You Need" paper did and which subsequent implementations quietly dropped.

**Both fixes together** are what the paper recommends as the complete solution. Fix 1 alone suppresses the exponential residual growth; Fix 2 alone suppresses the LN gradient amplification at early layers. Neither is sufficient on its own.

---

## Expected Outcome

With these changes, gradient norms at shallow layers should no longer be exponentially larger than at deep layers. We'd expect:
- Stable training at 24 and 30 layers with no loss spikes
- Ability to use larger learning rates (the paper shows stable models tolerate 3–10× larger LR, improving final quality)
- Consistent video quality that doesn't degrade after rare large gradient steps
