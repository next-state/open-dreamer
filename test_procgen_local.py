#!/usr/bin/env python3
"""
Local test script for Procgen integration (without Reactor runtime).
This helps verify the environment works before deploying to Reactor.
"""

import numpy as np
import logging
from dreamer.procgen_reactor import ProcgenVideoModel, ProcgenReactorConfig
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_procgen_model():
    """Test the Procgen model locally."""
    logger.info("Creating ProcgenVideoModel...")
    
    cfg = ProcgenReactorConfig(
        env_name="coinrun",
        num_levels=1,  # Use 1 level for testing
        start_level=0,
        distribution_mode="easy",
    )
    
    model = ProcgenVideoModel(
        fps=15,
        size=(64, 64),
        cfg=cfg,
    )
    
    logger.info("Model created successfully!")
    logger.info(f"Action dimension: {model.cfg.action_dim}")
    
    # Test environment reset
    logger.info("Testing environment...")
    obs = model.env.reset()
    logger.info(f"Environment reset. Observation keys: {obs.keys() if isinstance(obs, dict) else 'array'}")
    if isinstance(obs, dict):
        logger.info(f"RGB observation shape: {obs['rgb'].shape}")
    else:
        logger.info(f"Observation shape: {obs.shape}")
    
    # Test a few steps
    logger.info("Testing environment steps...")
    for i in range(10):
        action = np.random.randint(0, 7)  # Random action
        obs, reward, done, info = model.env.step(np.array([action]))
        logger.info(f"Step {i}: action={action}, reward={reward[0]:.2f}, done={done[0]}")
        
        if done[0]:
            logger.info("Episode finished, resetting...")
            obs = model.env.reset()
    
    # Clean up
    model.env.close()
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    test_procgen_model()
