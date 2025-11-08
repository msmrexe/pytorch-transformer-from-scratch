import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

# HuggingFace
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import logging

def causal_mask(size: int) -> torch.Tensor:
    """
    Creates a mask for the decoder to prevent attending to future tokens.
    Returns a (1, size, size) tensor where values above diagonal are 0.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Pre-calculate special token IDs for efficiency
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_pair = self.ds[idx]
        src_text = src_pair['translation'][self.src_lang]
        tgt_text = src_pair['translation'][self.tgt_lang]

        # Tokenize texts into integers
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate padding needed to reach seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 for SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 for SOS only

        # Fail fast if sentence is too long (should be filtered beforehand, but good safety check)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
             raise ValueError("Sentence is too long")

        # Create encoder input: [SOS] + tokens + [EOS] + [PAD]...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create decoder input: [SOS] + tokens + [PAD]...
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create labels (what we want to predict): tokens + [EOS] + [PAD]...
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check size
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # Encoder mask: (1, 1, seq_len) - unsqueeze to broadcast over heads and sequence
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # Decoder mask: (1, seq_len, seq_len) - combines padding mask and causal mask
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def get_all_sentences(ds, lang) -> Iterator[str]:
    """Yields all sentences in a specific language from the dataset."""
    for item in ds:
        yield item['translation'][lang]

def build_tokenizer(config: Dict[str, Any], ds, lang: str) -> Tokenizer:
    """
    Builds a WordLevel tokenizer or loads it if it already exists.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not tokenizer_path.exists():
        logging.info(f"Tokenizer not found. Building tokenizer for language: {lang}")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        logging.info(f"Tokenizer saved to {tokenizer_path}")
    else:
        logging.info(f"Loading existing tokenizer for language: {lang}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_ds(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    """
    Prepares the OpusBooks dataset for training.
    Downloads raw data, builds tokenizers, filters by length, and creates DataLoaders.
    """
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Filter dataset by max sequence length to avoid truncation issues
    # (A simplistic but effective approach for this scale)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    logging.info(f"Max length of source sentence: {max_len_src}")
    logging.info(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # Validation batch size is 1 for easier manual inspection of translations during training
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
