# Dreamer data pipeline

This package contains the Grain and ArrayRecord data path used by tokenizer
training, offline tokenization, dynamics training, and FVD evaluation.

The top-level README uses one concrete setup for the examples: fixed 256-frame
Minecraft/VPT-style videos, raw shards named `shard-00000.array_record`,
tokenized output written with 5,000 records per shard, and dynamics training
with 64-frame short chunks packed into 256-frame sequences. Those values are
examples, not global defaults.

## Raw Minecraft/VPT records

Raw Minecraft/VPT data is stored as ArrayRecord shards under one directory:

```text
/path/to/mp4-arrayrecords/
  shard-00000.array_record
  shard-00001.array_record
  ...
```

Each ArrayRecord entry is a pickled Python dict:

```python
{
    "video": mp4_bytes,                  # bytes for one MP4 clip
    "video_shape": (T, H, W, C),          # used for length filtering
    "actions": [action_dict_0, ...],      # VPT-style action dicts, length T
    "source": "optional-id-or-path",
}
```

Use fixed-length records for the tokenization workflow. The tokenization script
decodes and batches full episodes, so every video in a batch must have the same
shape.

## Tokenized latent records

`scripts/tokenize_minecraft_dataset.py` reads raw MP4 records, encodes full
episodes with the tokenizer, and writes msgpack ArrayRecord shards:

```text
/path/to/tokenized_data/
  shard-00000.array_record
  shard-00001.array_record
  metadata/latent_stats.npz
```

Each tokenized record contains:

```python
{
    "latents": latent_array,      # (T, n_latents, d_bottleneck)
    "actions": actions_dict,      # action arrays aligned with T
    "source": "optional-id-or-path",
}
```

After tokenization, update
`configs/dataset/minecraft_vpt_latent.yaml` with the tokenized
`array_record_path`, the number of latent shards in `index_max`, and the
generated `latent_mean` and `latent_std` from `metadata/latent_stats.npz`.

## Example values

The common "magic numbers" in the README and configs are tied to the example
dataset and hardware budget:

| Value | Meaning | When to change it |
| --- | --- | --- |
| `index_max=1500` | Number of raw shards loaded from `array_record_path`. For Minecraft/VPT, the path builder expands this to `shard-00000.array_record` through `shard-01499.array_record`. | Set this to the number of contiguous raw shards you want to use. It is not a video count. |
| `index_max=89` | Number of tokenized latent shards loaded for dynamics training in the example command. | Set this to the actual number of `shard-*.array_record` files in your tokenized output directory. |
| `records_per_shard=5000` | Number of tokenized episodes written before rotating to the next latent shard. | Raise it for fewer output files; lower it for smaller files and easier partial reruns. |
| `B=8` | Global batch size across all JAX processes. | Tune for memory and throughput. It must be divisible by `jax.process_count()`. |
| `short_T=16`, `long_T=16` for tokenizer training | Train the tokenizer on 16-frame clips. | Increase for longer reconstruction windows if memory allows. This can be shorter than the full episode length used during tokenization. |
| `short_T=64`, `long_T=256` for dynamics training | Produce 256-step training batches. Short batches pack four independent 64-step chunks; long batches use real 256-step episodes. | Keep `long_T <= T` and `long_T % short_T == 0`. For shorter records, reduce `long_T`. |
| `long_ratio=0.1` | Approximate fraction of dynamics batches drawn from true `long_T` episodes instead of packed short chunks. | Raise it to train on long contexts more often; lower it if long batches are too expensive. |
| `ctx_length=4`, `horizon=240` | FVD generation uses 4 context frames and predicts 240 future frames. The dataloader needs `ctx_length + horizon` frames. | Ensure `ctx_length + horizon <= T` for your records. |
| `fvd_chunk_size=16` | FVD clips are split into 16-frame chunks after optional context trimming. With `horizon=240`, this gives 15 predicted-frame chunks per generated video. | The evaluated length must be divisible by this value, and each chunk must be at least 10 frames for I3D. |
| `H=360`, `W=640`, `padding_H=[4, 4]`, `padding_W=[0, 0]`, `patch_size=16` | Raw 360x640 frames are padded to 368x640 so both spatial dimensions divide cleanly into 16x16 patches. | Adjust these together when changing resolution or tokenizer patch size. |
| `dataset_mean`, `dataset_std` | Pixel statistics used to normalize raw video values. | Recompute these whenever the raw video distribution changes. |

## Required constraints

- Raw shards for Minecraft/VPT and latent datasets must be named contiguously as
  `shard-00000.array_record`, `shard-00001.array_record`, and so on.
- `index_max` must be greater than zero for Minecraft/VPT and latent datasets.
- Every training record must have at least `long_T` frames.
- FVD generation records must have at least `ctx_length + horizon` frames.
- Dynamics training requires `long_T % short_T == 0`.
- The global batch size `B` must be divisible by `jax.process_count()`.
- `H + padding_H[0] + padding_H[1]` must be divisible by `patch_size`.
- `W + padding_W[0] + padding_W[1]` must be divisible by `patch_size`.
- If `fvd_chunk_size` is set, the evaluated video length after context trimming
  must be divisible by `fvd_chunk_size`.

## Useful checks

Count shards before setting `index_max`:

```bash
find /path/to/mp4-arrayrecords -maxdepth 1 -name 'shard-*.array_record' | sort | wc -l
find /path/to/tokenized_data -maxdepth 1 -name 'shard-*.array_record' | sort | wc -l
```

Print tokenization statistics:

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
