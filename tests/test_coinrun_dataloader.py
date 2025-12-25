"""
test_dataloader.py

Minimal visual sanity-check for CoinRun `.array_record` datasets.

Loads one batch via `coinrun_data/dataloader.get_dataloader` and saves an annotated
GIF showing frames with per-timestep action/reward overlays.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import numpy as np

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from coinrun_data.dataloader import get_dataloader


def _to_scalar(x: Any) -> Any:
    """Best-effort scalar conversion for display."""
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.reshape(()).item()
        return x.tolist()
    return x

def _annotate_frame(
    frame: np.ndarray,
    line1: str,
    line2: str,
    font: Optional[ImageFont.ImageFont] = None,
    padding_height: int = 40,
) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    h, w, c = frame.shape
    # Create a new image with padding at the top.
    padded_img = Image.new("RGB", (w, h + padding_height), color=(0, 0, 0))
    # Paste the original frame below the padding.
    frame_img = Image.fromarray(frame)
    padded_img.paste(frame_img, (0, padding_height))

    draw = ImageDraw.Draw(padded_img)
    font = font or ImageFont.load_default()

    # Draw two lines of text in the padding area.
    pad_x = 4
    pad_y = 4
    draw.text((pad_x, pad_y), line1, fill=(255, 255, 255), font=font)
    # Get line height to position second line.
    bbox = draw.textbbox((0, 0), line1, font=font)
    line_height = bbox[3] - bbox[1]
    draw.text((pad_x, pad_y + line_height + 2), line2, fill=(255, 255, 255), font=font)
    return np.asarray(padded_img)


def _create_grid_frame(
    annotated_frames: list[np.ndarray],
    grid_width: int,
    grid_height: int,
) -> np.ndarray:
    """
    Arrange annotated frames into a grid.
    
    Args:
        annotated_frames: List of annotated frames, length should be grid_width * grid_height
        grid_width: Number of columns in the grid
        grid_height: Number of rows in the grid
    
    Returns:
        Combined grid image as numpy array
    """
    if len(annotated_frames) != grid_width * grid_height:
        raise ValueError(
            f"Expected {grid_width * grid_height} frames, got {len(annotated_frames)}"
        )
    
    # Get dimensions from first frame
    frame_h, frame_w, frame_c = annotated_frames[0].shape
    
    # Create grid image
    grid_img = Image.new("RGB", (grid_width * frame_w, grid_height * frame_h), color=(0, 0, 0))
    
    for idx, frame_arr in enumerate(annotated_frames):
        row = idx // grid_width
        col = idx % grid_width
        frame_img = Image.fromarray(frame_arr)
        grid_img.paste(frame_img, (col * frame_w, row * frame_h))
    
    return np.asarray(grid_img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-gif", type=str, default="dataloader_debug.gif")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--grid-width", type=int, required=True)
    parser.add_argument("--grid-height", type=int, required=True)
    parser.add_argument("--image-h", type=int, default=64)
    parser.add_argument("--image-w", type=int, default=64)
    parser.add_argument("--image-c", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-buffer-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    
    if args.global_batch_size != args.grid_width * args.grid_height:
        raise ValueError(
            f"batch_size ({args.global_batch_size}) must equal grid_width * grid_height "
            f"({args.grid_width} * {args.grid_height} = {args.grid_width * args.grid_height})"
        )

    dl = get_dataloader(
        array_record_paths=args.data_dir,
        seq_len=args.seq_len,
        global_batch_size=args.global_batch_size,
        image_h=args.image_h,
        image_w=args.image_w,
        image_c=args.image_c,
        num_workers=args.num_workers,
        prefetch_buffer_size=args.prefetch_buffer_size,
        seed=args.seed,
    )

    it = iter(dl)
    batch = next(it)
    if not isinstance(batch, dict) or "videos" not in batch:
        raise ValueError(f"Unexpected batch type/shape: {type(batch)} keys={getattr(batch, 'keys', lambda: [])()}")

    videos = batch["videos"]
    if not isinstance(videos, np.ndarray):
        videos = np.asarray(videos)
    if videos.ndim != 5:
        raise ValueError(f"Expected videos with shape (B,T,H,W,C), got {videos.shape}")
    
    batch_size, seq_len = videos.shape[0], videos.shape[1]
    if batch_size != args.global_batch_size:
        raise ValueError(
            f"Expected batch_size={args.global_batch_size}, got {batch_size} from dataloader"
        )

    actions = batch.get("actions", None)
    rewards = batch.get("rewards", None)
    if actions is not None:
        actions = np.asarray(actions)
    if rewards is not None:
        rewards = np.asarray(rewards)

    # Count videos with reward > 0
    if rewards is not None:
        videos_with_positive_reward = 0
        for sample_idx in range(batch_size):
            video_rewards = rewards[sample_idx]  # Shape: (seq_len,)
            if np.any(video_rewards > 0):
                videos_with_positive_reward += 1
        print(f"Number of videos with reward > 0: {videos_with_positive_reward} / {batch_size}")
    else:
        print("Rewards not available in batch")

    # Try to load a smaller font, fall back to default if not available.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=14)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    # For each timestep, create a grid frame
    grid_frames = []
    for t in range(seq_len):
        annotated_cells = []
        for sample_idx in range(batch_size):
            frame = videos[sample_idx, t].copy()
            action_t = _to_scalar(actions[sample_idx, t]) if actions is not None else "N/A"
            reward_t = _to_scalar(rewards[sample_idx, t]) if rewards is not None else "N/A"
            
            # Add visual cue if reward > 0: brighten the frame
            if rewards is not None and reward_t != "N/A":
                try:
                    reward_val = float(reward_t)
                    if reward_val > 0:
                        # Brighten the frame by multiplying pixel values
                        frame = frame.astype(np.float32)
                        frame = frame * 1.5  # Make it 50% brighter
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                except (ValueError, TypeError):
                    pass  # If reward_t can't be converted to float, skip visual cue
            
            line1 = f"{t},{action_t}"
            line2 = f"{reward_t}"
            annotated_cells.append(_annotate_frame(frame, line1=line1, line2=line2, font=font))
        
        grid_frame = _create_grid_frame(annotated_cells, args.grid_width, args.grid_height)
        grid_frames.append(grid_frame)

    out_path = args.out_gif
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    imageio.mimsave(out_path, grid_frames, fps=args.fps)
    print(f"Wrote {out_path} ({len(grid_frames)} frames, grid={args.grid_width}x{args.grid_height})")


if __name__ == "__main__":
    main()