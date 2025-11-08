import sys
from pathlib import Path
import logging

# Add project root to sys.path to enable importing from src regardless of CWD
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import get_config, get_args
from src.utils import setup_logging
from src.engine import train_model

def main():
    # 1. Setup centralized logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 2. Load configuration
    base_config = get_config()
    config = get_args(base_config)
    
    # 3. Announce start and log configuration for reproducibility
    logger.info("=" * 50)
    logger.info("Starting Transformer Training Run")
    logger.info("=" * 50)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"{key:>20}: {value}")
    logger.info("=" * 50)

    # 4. Launch training engine
    try:
        train_model(config)
        logger.info("Training completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
