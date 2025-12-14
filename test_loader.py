#%%
import jax
import jax.numpy as jnp
from dreamer.data import DatasetConfig, make_iterator
import imageio

def test_dataloader():
    print("Initializing DatasetConfig...")
    cfg = DatasetConfig(
        source="custom",
        array_record_path="datasets/coinrun_episodes/train",
        B=4,
        T=16,
        H=64,
        W=64,
        C=3,
    )
    
    print(f"Config: {cfg}")
    
    print("Creating iterator...")
    iterator = make_iterator(cfg)
    
    rng = jax.random.PRNGKey(0)
    
    print("Fetching first batch...")
    batch = None
    for batch in iterator:
        break
    rewards = batch['rewards']
    video = batch['videos']
    actions = batch['actions']
    
    print("Batch shapes:")
    print(f"Video: {video.shape} (Type: {video.dtype})")
    print(f"Actions: {actions.shape} (Type: {actions.dtype})")
    print(f"Rewards: {rewards.shape} (Type: {rewards.dtype})")
    
    # Basic assertions
    assert video.shape == (cfg.B, cfg.T, cfg.H, cfg.W, cfg.C)
    # Actions should be (B, T) because make_bouncing_square_iterator prepends a dummy action
    assert actions.shape == (cfg.B, cfg.T) 
    assert rewards.shape == (cfg.B, cfg.T)
    
    print("Value checks:")
    print(f"Video range: [{video.min()}, {video.max()}]")
    print(f"Actions unique: {jnp.unique(actions)}")
    # rewards[0] is NaN
    print(f"Rewards[0,0] (should be NaN): {rewards[0,0]}")
    
    # Save a gif for visual inspection
    print("Saving test_batch.gif...")
    # Frames: (B, T, H, W, C) -> take first batch element: (T, H, W, C)
    frames = video[0] 
    frames_uint8 = (frames * 255).astype(jnp.uint8)
    imageio.mimsave("test_batch.gif", frames_uint8, fps=10)
    print("Saved test_batch.gif")

    print("Test passed!")
#%%
if __name__ == "__main__":
    test_dataloader()

# %%
