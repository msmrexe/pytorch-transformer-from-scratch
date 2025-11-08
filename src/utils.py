import logging
import shutil
from pathlib import Path
from typing import Union, Dict, Any

def setup_logging(log_dir: Union[str, Path] = "logs", log_filename: str = "transformer.log"):
    """
    Configures the root logger to write to console (INFO) and a file (DEBUG).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    full_log_path = log_path / log_filename

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(full_log_path, mode='w') # 'w' to overwrite each run, 'a' to append

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(levelname)s: %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    # Remove existing handlers to avoid duplicate logs in notebooks/re-runs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    logging.info(f"Logging setup complete. Log file: {full_log_path}")

def get_weights_file_path(config: Dict[str, Any], epoch: str) -> str:
    """
    Constructs the full path for a model checkpoint.
    """
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(model_folder) / model_filename)

def get_latest_weights_file_path(config: Dict[str, Any]) -> Union[str, None]:
    """
    Finds the latest model checkpoint in the model folder based on numerical epoch in filename.
    """
    model_folder = Path(config["model_folder"])
    model_basename = config["model_basename"]
    
    if not model_folder.exists():
        return None

    items = [str(x) for x in model_folder.glob(f"{model_basename}*.pt")]
    if not items:
        return None
    
    # Sort by epoch number assuming format "{basename}{epoch}.pt"
    # This is a simplistic sort, might need robust regex if naming changes significantly
    try:
        latest = max(items, key=lambda x: int(Path(x).stem.replace(model_basename, "")))
        return latest
    except ValueError:
        # Fallback if standard naming failed
        return max(items, key=os.path.getctime)

def get_console_width() -> int:
    """
    Returns the width of the console for pretty printing.
    """
    return shutil.get_terminal_size((80, 20)).columns
