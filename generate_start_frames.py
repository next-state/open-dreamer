#!/usr/bin/env python3
"""
Throwaway script to generate canonical starting frames for reactor initialization.
Creates a set of reset frames from CoinRun at different levels.
"""

import imageio
import numpy as np
from pathlib import Path
from procgen import ProcgenEnv

def generate_start_frames(
    env_name: str = "coinrun",
    num_levels: int = 10,
    output_dir: str = "assets/start_frames",
    context_length: int = 16,
):
    """
    Generate starting frames from CoinRun environment resets.
    
    Args:
        env_name: Procgen environment name
        num_levels: Number of different levels to sample
        output_dir: Directory to save frames
        context_length: Number of frames to capture per level
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_levels} starting sequences of {context_length} frames each...")
    
    for level_idx in range(num_levels):
        print(f"Level {level_idx}...", end=" ", flush=True)
        
        # Create environment at specific level
        env = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            start_level=level_idx,
            num_levels=1,
            distribution_mode="easy",
        )
        
        # Reset and get initial observation
        obs = env.reset()
        
        frames = []
        for frame_idx in range(context_length):
            # Get current frame
            frame = obs["rgb"][0]  # Shape: (H, W, C)
            frames.append(frame)
            
            # Take a no-op action to get next frame
            obs, rew, done, info = env.step(np.array([4]))  # Action 4 = no movement in CoinRun
        
        # Stack frames: (T, H, W, C)
        frames = np.stack(frames, axis=0)
        
        # Save individual starting frame
        frame_path = output_path / f"level_{level_idx:03d}_frame0.png"
        imageio.imwrite(frame_path, frames[0])
        
        # Save the full sequence as numpy
        npy_path = output_path / f"level_{level_idx:03d}.npy"
        np.save(npy_path, frames)
        
        env.close()
        print(f"✓ Saved to {frame_path.name}, {npy_path.name}")
    
    # Also create a "canonical" single frame by averaging first frames
    print("\nCreating canonical average frame...")
    all_first_frames = []
    for level_idx in range(num_levels):
        npy_path = output_path / f"level_{level_idx:03d}.npy"
        frames = np.load(npy_path)
        all_first_frames.append(frames[0])
    
    avg_frame = np.mean(all_first_frames, axis=0).astype(np.uint8)
    canonical_path = output_path / "canonical_start.png"
    imageio.imwrite(canonical_path, avg_frame)
    print(f"✓ Saved canonical frame to {canonical_path.name}")
    
    print(f"\nDone! Generated {num_levels} starting sequences in {output_path}")
    print(f"Frame shape: {frames[0].shape}")
    print(f"Sequence shape: {frames.shape}")
    print(f"\nYou can now use these in reactor.py like:")
    print(f"  init_frames = np.load('assets/start_frames/level_000.npy')")
    print(f"  init_frames = jnp.array(init_frames)[None]  # Add batch dim")


if __name__ == "__main__":
    generate_start_frames(
        env_name="coinrun",
        num_levels=10,
        output_dir="assets/start_frames",
        context_length=16,
    )
