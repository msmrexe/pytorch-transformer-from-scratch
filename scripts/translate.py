import sys
import torch
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import get_config, get_args
from src.model import build_transformer
from src.data_loader import build_tokenizer
from src.engine import greedy_decode
from src.utils import setup_logging

# Mock dataset just for tokenizer loading compatibility
class MockDataset:
    def __iter__(self): return iter([])

def translate_sentence(text: str, model, tokenizer_src, tokenizer_tgt, device, seq_len):
    """
    Preprocesses a single sentence, runs inference, and decodes the output.
    """
    model.eval()
    
    with torch.no_grad():
        # 1. Tokenize and encode the source text
        sos_token = tokenizer_tgt.token_to_id('[SOS]')
        eos_token = tokenizer_tgt.token_to_id('[EOS]')
        pad_token = tokenizer_tgt.token_to_id('[PAD]')

        encoder_input_tokens = tokenizer_src.encode(text).ids
        
        # 2. Pad the sequence to the fixed seq_len expected by the model
        num_padding = seq_len - len(encoder_input_tokens) - 2 # -2 for SOS and EOS
        if num_padding < 0:
             raise ValueError(f"Sentence is too long. Max length is {seq_len - 2} tokens.")

        encoder_input = torch.cat(
            [
                torch.tensor([sos_token], dtype=torch.int64),
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                torch.tensor([eos_token], dtype=torch.int64),
                torch.tensor([pad_token] * num_padding, dtype=torch.int64),
            ],
            dim=0,
        ).to(device)

        # 3. Create the encoder mask (mask out padding tokens)
        # (1, 1, seq_len)
        encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

        # 4. Run inference using greedy decoding
        # Add batch dimension: (seq_len) -> (1, seq_len)
        model_output_tokens = greedy_decode(
            model, encoder_input.unsqueeze(0), encoder_mask, 
            tokenizer_src, tokenizer_tgt, seq_len, device
        )

        # 5. Decode the resulting tokens back into text
        translated_text = tokenizer_tgt.decode(model_output_tokens.detach().cpu().numpy(), skip_special_tokens=True)
    
    return translated_text

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    base_config = get_config()
    config = get_args(base_config)

    # Validate required arguments for inference
    if not config['inference_model_path']:
        logger.error("Please provide a path to a model checkpoint using --model_path")
        sys.exit(1)
    if not config['inference_text']:
        logger.error("Please provide text to translate using --text")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on device: {device}")

    # Load Tokenizers (Using MockDataset because we just need to load from file, not train)
    # If tokenizers don't exist, this will fail, which is expected for pure inference mode.
    try:
        tokenizer_src = build_tokenizer(config, MockDataset(), config['lang_src'])
        tokenizer_tgt = build_tokenizer(config, MockDataset(), config['lang_tgt'])
    except Exception as e:
        logger.error("Could not load tokenizers. Ensure you have trained the model or have 'tokenizer_{lang}.json' files in the root.")
        raise e

    # Build Model Architecture
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'], config['seq_len'],
        d_model=config['d_model']
    ).to(device)

    # Load Model Weights
    logger.info(f"Loading model weights from: {config['inference_model_path']}")
    state = torch.load(config['inference_model_path'], map_location=device)
    model.load_state_dict(state['model_state_dict'])

    # Run Translation
    input_text = config["inference_text"]
    logger.info(f"Translating...")
    output_text = translate_sentence(input_text, model, tokenizer_src, tokenizer_tgt, device, config['seq_len'])

    # Print Final Result clearly
    print("\n" + "="*40)
    print(f"SOURCE ({config['lang_src']}): {input_text}")
    print(f"TARGET ({config['lang_tgt']}): {output_text}")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
