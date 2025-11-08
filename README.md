# PyTorch Transformer (from Scratch) for Neural Machine Translation

This repository provides a comprehensive, from-scratch implementation of the original "Attention Is All You Need" Transformer model using PyTorch. Developed for an M.S. Machine Learning course, this project goes beyond a simple model definition; it's a complete, end-to-end pipeline for **Neural Machine Translation (NMT)**.

It includes modular, professional-grade code for data loading, tokenization, training, inference, and logging, all focused on translating English to Italian using the Hugging Face `OpusBooks` dataset.

## Features

* **Complete Architecture:** Full Encoder-Decoder Transformer model built from scratch using `nn.Module`.
* **Modern PyTorch:** Clean, modular, and heavily-typed Python code.
* **Core Components:** Implements Multi-Head Attention, Positional Encodings, Pre-Layer-Normalization, and Residual Connections.
* **Precise Masking:** Correctly implements both **padding masks** (for encoder and decoder) and a **causal (look-ahead) mask** (for the decoder).
* **End-to-End Pipeline:** Uses Hugging Face `datasets` to load the corpus and `tokenizers` to build a Word-Level vocabulary from scratch.
* **Full Workflow:** Provides separate, executable scripts for training (`scripts/train.py`) and inference (`scripts/translate.py`).
* **Logging & Monitoring:** Includes robust logging to both console and file (`logs/`) via Python's `logging` module, plus TensorBoard integration (`runs/`) for tracking training loss.

## Key Concepts & Architecture

This repository implements the architecture from the "Attention Is All You Need" paper. Here is a conceptual breakdown from the ground up.

### 1. The Problem: Why RNNs Aren't Enough

Traditional Seq2Seq models used RNNs (like LSTMs or GRUs). These models read a sentence word-by-word, building a "context vector" (the final hidden state) that tries to summarize the whole sentence. This has two major flaws:
1.  **Information Bottleneck:** The entire meaning of a 50-word sentence must be compressed into one fixed-size vector. This is incredibly lossy.
2.  **Sequential Computation:** It's slow. You can't process the 10th word until you've processed the 9th, making parallelization on modern GPUs impossible.

### 2. The Core Idea: Self-Attention

The Transformer's solution is **self-attention**. Instead of a bottleneck, it allows every word to look at and interact with every other word in the sequence simultaneously.

It does this using **Query (Q)**, **Key (K)**, and **Value (V)** vectors.

* **Analogy:** Think of retrieving a file from a cabinet.
    * **Query (Q):** A sticky note with "report on Q3 sales" (what I'm *looking for*).
    * **Key (K):** The labels on the file folders ("Q1 Sales," "Q2 Sales," "Q3 Sales").
    * **Value (V):** The actual contents of the folders.

You compare your **Query** ("Q3 sales") to every **Key** to find the best match (the highest "attention score"). You then pull out the **Value** (the contents) of that matching folder.

Self-attention does this with vectors. For *every* token in the sequence:
1.  It generates a Q, K, and V vector for that token.
2.  It compares its **Q** vector to *every other token's* **K** vector to get "attention scores" (how relevant they are to each other).
3.  These scores are scaled (divided by $\sqrt{d_k}$) and put through a `softmax` to turn them into weights that sum to 1.
4.  It multiplies these weights by each token's **V** vector, creating a new vector that is a weighted sum of all other values in the sequence.

The full formula for **Scaled Dot-Product Attention** is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This process happens *in parallel* for every token, making it much faster than an RNN.

### 3. The Full Architecture

#### **Input: Embeddings & Positional Encoding**

Since the self-attention mechanism is order-agnostic (it sees a "bag" of tokens), we must inject information about the token's position. We do this by adding a **Positional Encoding** vector to the word embedding. This vector is generated using fixed `sin` and `cos` functions:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

#### **The Encoder Stack (N=6)**

The Encoder's job is to read the input (English) sentence and build a rich, contextual representation of it. Each `EncoderBlock` has two sub-layers:

1.  **Multi-Head Attention:** This is the self-attention mechanism run *h* (e.g., 8) times in parallel. Each "head" learns different types of relationships. The results are concatenated and projected.
2.  **Feed-Forward Network:** A simple two-layer, position-wise fully-connected network. This processes each token's representation *independently*.

* Each sub-layer is wrapped in a **Residual Connection** and **Layer Normalization**. This implementation uses **Pre-LN**, where normalization is applied *before* the sub-layer: `x = x + Sublayer(Norm(x))`, which is known for more stable training.

#### **The Decoder Stack (N=6)**

The Decoder's job is to generate the output (Italian) sentence, token by token. Each `DecoderBlock` has *three* sub-layers:

1.  **Masked Multi-Head Attention:** This is self-attention on the *target* (Italian) sentence as it's being generated. It's **masked** (a "look-ahead mask") to prevent a token at position `i` from attending to tokens at positions `j > i`. This is crucial, as we can't let the model "cheat" by seeing the future.
2.  **Cross-Attention:** This is the most important part for translation. The **Q** vector comes from the decoder, but the **K and V vectors come from the *Encoder's output***. This is the critical step where the decoder *conditions* its output on the encoder's understanding of the source sentence.
3.  **Feed-Forward Network:** Identical to the encoder's.

* All three sub-layers also use the same Pre-LN `Add & Norm` wrappers.

#### **Final Projection Layer**

After the final `DecoderBlock`, a single Linear layer and a `LogSoftmax` function act as a classifier. The Linear layer projects the final `d_model` vector into a vector the size of the target vocabulary (e.g., ~30,000 words). The `LogSoftmax` turns this into log-probabilities, which are used with `CrossEntropyLoss` to calculate the training loss.

## Implementation Details

This repository's code is structured to be modular and clear, mapping directly to the concepts above.

* `src/model.py`: Contains all the `nn.Module` building blocks. You'll find classes like `MultiHeadAttentionBlock`, `FeedForwardBlock`, `PositionalEncoding`, `ResidualConnection` (implemented as Pre-LN), and the main `Transformer` module that assembles them.
* `src/data_loader.py`: Handles the entire data pipeline.
    1.  `get_ds` loads the raw text from Hugging Face `datasets`.
    2.  `build_tokenizer` trains a `WordLevel` tokenizer from scratch and saves it to disk.
    3.  `BilingualDataset` is a PyTorch `Dataset` that, for each item, takes raw text, tokenizes it, adds `[SOS]`, `[EOS]`, and `[PAD]` tokens, and most importantly, creates the `encoder_mask` and `decoder_mask` required by the model.
* `src/engine.py`: Contains the core training and inference logic.
    1.  `train_one_epoch` handles the main training loop (forward pass, loss calculation, backpropagation).
    2.  `run_validation` iterates over the validation set and calls...
    3.  `greedy_decode`, which performs live inference by taking one `[SOS]` token, feeding it to the decoder, and repeatedly taking the most probable next token until `[EOS]` is generated.
* `scripts/train.py`: The main entry point to start a training run. It initializes logging, parses command-line args, builds the model, and calls `train_model` from the engine.
* `scripts/translate.py`: The inference entry point. It loads a trained model checkpoint, loads the tokenizers, and uses the `greedy_decode` engine function to translate a single sentence provided via the command line.

---

## Project Structure

```
pytorch-transformer-from-scratch/
├── .gitignore              # Ignores Python cache, data, logs, models, etc.
├── LICENSE                 # MIT License file
├── README.md               # This file
├── requirements.txt        # Project dependencies (torch, datasets, tokenizers)
├── run_translation.ipynb   # Guided notebook to run training and inference
├── data/
│   └── .gitkeep            # Stores trained tokenizers (e.g., tokenizer\_en.json)
├── logs/
│   └── .gitkeep            # Stores training log files (e.g., transformer.log)
├── models/
│   └── .gitkeep            # Stores saved model checkpoints (e.g., tmodel\_01.pt)
├── runs/
│   └── .gitkeep            # Stores TensorBoard logs
├── scripts/
│   ├── train.py            # Main script to start a training run
│   └── translate.py        # Main script to run inference on a sentence
└── src/
    ├── __init__.py         # Makes 'src' a Python package
    ├── config.py           # Manages config and command-line arguments (argparse)
    ├── data_loader.py      # Handles dataset loading, tokenizing, and BilingualDataset class
    ├── engine.py           # Core training loop, validation loop, and inference logic
    ├── model.py            # All nn.Module classes for the Transformer architecture
    └── utils.py            # Utility functions (logging setup, path management)
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-transformer-from-scratch.git
    cd pytorch-transformer-from-scratch
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model:**
    Run the training script. This will download the dataset, build the tokenizers (saving them to `data/`), and start training. All checkpoints will be saved to `models/`.
    ```bash
    # Train for 10 epochs with a batch size of 16
    python scripts/train.py --num_epochs 10 --batch_size 16
    ```
    * You can monitor the training progress in real-time using TensorBoard:
        ```bash
        tensorboard --logdir=runs
        ```

4.  **Run Inference (Translation):**
    Once you have a trained model checkpoint, you can use the `translate.py` script.
    ```bash
    # Example using the model saved from the 10th epoch (which is 0-indexed '09')
    # Check your 'models/' folder for the exact filename!
    python scripts/translate.py --model_path "models/tmodel_09.pt" --text "I love machine learning."
    ```

    **Example Output:**
    ```
    ========================================
    SOURCE (en): I love machine learning.
    TARGET (it): Amo l'apprendimento automatico.
    ========================================
    ```

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
