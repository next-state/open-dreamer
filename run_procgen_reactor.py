#!/usr/bin/env python3
"""
Entry point for running Procgen CoinRun with Reactor.

Usage:
    python run_procgen_reactor.py
"""

import logging
from dreamer.procgen_reactor import ProcgenVideoModel, ProcgenReactorConfig

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Procgen CoinRun Reactor...")
    
    # Configure the environment
    cfg = ProcgenReactorConfig(
        env_name="coinrun",
        num_levels=0,  # 0 = infinite procedural levels
        start_level=0,
        distribution_mode="easy",  # "easy", "hard", or "exploration"
    )
    
    # Create the video model
    # The Reactor runtime will automatically call this and manage sessions
    model = ProcgenVideoModel(
        fps=15,  # CoinRun runs at 15 FPS
        size=(64, 64),  # Native CoinRun resolution
        cfg=cfg,
    )
    
    logger.info("Procgen CoinRun Reactor initialized and ready!")
    logger.info("The server should now be accessible via the Reactor client.")
    
    # The reactor_runtime will handle the server lifecycle
    # This script just needs to keep the model instance alive


if __name__ == "__main__":
    main()
