"""Test script to verify CoinRun action mappings."""

from procgen import ProcgenEnv
import numpy as np

# Create environment
env = ProcgenEnv(
    num_envs=1,
    env_name="coinrun",
    num_levels=1,
    start_level=0,
    distribution_mode="easy",
)

# Reset to get initial state
obs = env.reset()
print("Testing CoinRun actions...")
print("=" * 50)

# Test each action
actions_to_test = [
    (0, "No movement"),
    (1, "Right (D key)"),
    (2, "Left (A key)"),
    (3, "Jump/Up (W key)"),
    (4, "Right-Jump (E key)"),
    (5, "Left-Jump (Q key)"),
    (6, "Down (S key)"),
]

for action_idx, description in actions_to_test:
    # Reset environment
    obs = env.reset()
    initial_frame = obs['rgb'][0].copy()
    
    # Take action for a few steps
    for _ in range(5):
        obs, reward, done, info = env.step(np.array([action_idx]))
    
    final_frame = obs['rgb'][0]
    
    # Simple check: did the frame change?
    frame_changed = not np.array_equal(initial_frame, final_frame)
    
    print(f"Action {action_idx}: {description}")
    print(f"  Frame changed: {frame_changed}")
    if reward[0] != 0:
        print(f"  Reward: {reward[0]}")
    print()

env.close()
print("Test complete!")
