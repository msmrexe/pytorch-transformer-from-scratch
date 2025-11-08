import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import logging

from src.data_loader import get_ds, causal_mask
from src.model import build_transformer
from src.utils import get_weights_file_path, get_latest_weights_file_path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Performs greedy decoding to generate a translation.
    Starts with [SOS] and repeatedly picks the token with the highest probability.
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step of decoding
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token probability
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], 
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, writer, global_step):
    model.eval()
    count = 0

    # Use a fixed width for pretty printing
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Log to console
            logging.info("-" * console_width)
            logging.info(f"{f'SOURCE: ':>12}{source_text}")
            logging.info(f"{f'TARGET: ':>12}{target_text}")
            logging.info(f"{f'PREDICTED: ':>12}{model_out_text}")

            if writer:
                # Log to TensorBoard
                writer.add_text("Validation/Source", source_text, global_step)
                writer.add_text("Validation/Target", target_text, global_step)
                writer.add_text("Validation/Predicted", model_out_text, global_step)
                writer.flush()

            # Only print 2 examples to avoid spamming logs
            if count == 2:
                break

def train_one_epoch(model, train_dataloader, optimizer, loss_fn, device, writer, epoch, global_step):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    
    for batch in batch_iterator:
        encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

        # Run the tensors through the transformer
        encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
        proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

        # Compare the output with the label
        label = batch['label'].to(device) # (B, seq_len)

        # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
        
        # Update progress bar and tensorboard
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        if writer:
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

        # Backpropagate
        loss.backward()

        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        
    return global_step

def train_model(config: Dict[str, Any]):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Ensure model folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Prepare data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Build Model
    model = build_transformer(
        tokenizer_src.get_vocab_size(), 
        tokenizer_tgt.get_vocab_size(), 
        config['seq_len'], config['seq_len'], 
        d_model=config['d_model']
    ).to(device)

    # Setup Training
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # Ignore padding tokens when calculating loss and use label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Initialize Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Preload Model Logic
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_latest_weights_file_path(config) if config['preload'] == 'latest' else get_weights_file_path(config, config['preload'])
        if model_filename:
            logging.info(f"Preloading model {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
             logging.warning(f"Could not find model to preload: {config['preload']}")

    # Training Loop
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        logging.info(f"Starting Epoch {epoch}")
        
        global_step = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device, writer, epoch, global_step)
        
        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, writer, global_step)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        logging.info(f"Saved model to {model_filename}")

    writer.close()
