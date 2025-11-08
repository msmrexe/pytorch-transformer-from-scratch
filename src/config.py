import argparse
from pathlib import Path
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """
    Returns the default configuration parameters for the Transformer model and training loop.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "d_ff": 2048,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "lang_src": "en",
        "lang_tgt": "it",
        "data_dir": Path("data"),
        "model_folder": Path("models"),
        "model_basename": "tmodel_",
        "preload": None,  # Can be 'latest', specific epoch number, or None
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_args(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses command-line arguments and overrides the base configuration.
    """
    parser = argparse.ArgumentParser(description="Train or Run Inference on the Transformer Model.")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=base_config["num_epochs"], help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=base_config["batch_size"], help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=base_config["lr"], help="Learning rate.")
    parser.add_argument("--seq_len", type=int, default=base_config["seq_len"], help="Maximum sequence length.")
    parser.add_argument("--preload", type=str, default=base_config["preload"], help="Epoch to preload (e.g., 'latest' or '10').")

    # Inference arguments
    parser.add_argument("--model_path", type=str, default=None, help="Path to a specific model checkpoint for inference.")
    parser.add_argument("--text", type=str, default=None, help="Source text to translate.")

    args = parser.parse_args()

    # Update base config with args if they are provided
    config = base_config.copy()
    if args.num_epochs is not None: config["num_epochs"] = args.num_epochs
    if args.batch_size is not None: config["batch_size"] = args.batch_size
    if args.lr is not None: config["lr"] = args.lr
    if args.seq_len is not None: config["seq_len"] = args.seq_len
    if args.preload is not None: config["preload"] = args.preload

    # Add inference specific args to config temporarily for scripts that need them
    config["inference_model_path"] = args.model_path
    config["inference_text"] = args.text

    return config
