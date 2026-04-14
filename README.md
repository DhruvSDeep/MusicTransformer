# MusicTransformer

A decoder-only transformer that **generates MIDI music from scratch**, built entirely in PyTorch. The pipeline reads `.midi` files, converts them into a compact token sequence encoding pitch, velocity, and duration, trains a GPT-style language model on those sequences, and autoregressively samples new compositions that are written back out as playable MIDI.

---

## Highlights

- **End-to-end MIDI pipeline** — raw `.midi` in, generated `.midi` out. No external music-ML libraries needed beyond `mido` for MIDI I/O.
- **Structured music tokenization** — each note becomes a single token encoding pitch (0–127), quantized velocity (1–31), and quantized duration (1–64 steps). Rests are represented as explicit time-shift tokens.
- **Rare-token pruning** — tokens that appear fewer than a configurable threshold are eliminated and the vocabulary is compactly remapped, dramatically shrinking the effective vocab size from a theoretical ~254 K down to only the tokens that actually appear in the dataset.
- **From-scratch transformer** — multi-head causal self-attention, sinusoidal positional encoding, pre-norm residual blocks, GELU feed-forward layers, and weight-tied embeddings.
- **Generation parameter search** — included utility sweeps over temperature × top-k grids and exports each variant as a separate MIDI file.

---

## Architecture

| Component | Detail |
|---|---|
| Type | Decoder-only (GPT-style) |
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 6 |
| Feed-forward dimension | 1024 |
| Context length | 512 tokens |
| FF activation | GELU |
| Normalization | Pre-LayerNorm |
| Weight tying | Embedding ↔ output head |

---

## MIDI Tokenization

Music is represented as a flat sequence of tokens. Each note in a MIDI file is encoded as:

```
P{pitch}_V{velocity}_D{duration}
```

where `pitch` is the raw MIDI note (0–127), `velocity` is quantized into 32 bins (1–31 after excluding silent), and `duration` is quantized into time steps (1–64, capped at 4 beats). Time gaps between notes are encoded as rest tokens:

```
REST_T{steps}
```

Rests longer than 32 steps are split into multiple `REST_T32` tokens. All tokens are then mapped to integer IDs via a deterministic formula, with special tokens `<PAD>` (0), `<SOS>` (1), and `<EOS>` (2) reserved.

### Vocabulary Pruning

The raw token space is ~254 K IDs (128 pitches × 31 velocities × 64 durations + 32 rests + 3 special). Most of these combinations never appear in real music. The `elim_tokens` module counts every token's occurrences across the dataset, eliminates those below a threshold (default: 6), and builds a compact remapping dictionary. This keeps training efficient and the embedding matrix tractable.

---

## Project Structure

```
MusicTransformer/
├── src/
│   ├── main.py               # Training entry point — tokenize MIDI, prune vocab, train
│   ├── transformerLogic.py    # Model architecture, training loop, generation (top-k sampling)
│   ├── dataLogic.py           # MIDI tokenization/detokenization, integer encoding, Dataset class
│   ├── elim_tokens.py         # Rare-token elimination and vocabulary remapping
│   ├── generator.py           # Load a checkpoint, seed from a random song, export generated MIDI
│   ├── paramsearch.py         # Grid search over temperature × top-k, exports each as a MIDI file
│   └── testLoss.py            # Evaluate cross-entropy loss on the full dataset
├── data/                      # MIDI corpus + serialized vocab/sequence files (gitignored)
├── checkpoints/               # Saved model weights (gitignored)
├── outputs/                   # Generated MIDI files (gitignored)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (CUDA recommended)
- [mido](https://mido.readthedocs.io/)

```bash
pip install torch mido
```

### Prepare Your Data

Place `.midi` files anywhere under `data/` (subdirectories are searched recursively). On the first training run, the pipeline will tokenize every file, build the occurrence dictionary, prune rare tokens, and cache the results as `.pkl` files.

### Train

```bash
cd src
python main.py
```

Training hyperparameters (configured in `transformerLogic.py`):

| Parameter | Value |
|---|---|
| Batch size | 16 |
| Learning rate | 1.2 × 10⁻³ |
| Weight decay | 0.01 |
| Warm-up | 3 epochs |
| LR schedule | Cosine annealing (after warm-up) |
| Gradient clipping | Max norm 1.0 |
| Epochs | 150 |

Checkpoints are saved to `checkpoints/` every 10 epochs.

### Generate Music

```bash
cd src
python generator.py
```

This script loads a trained checkpoint, picks a random song from the dataset as a seed (first 100 tokens), generates a continuation, and writes the result to `outputs/trial1.midi`. Generation uses a two-pass strategy: generate 1024 tokens, take the tail as a new seed, and generate another 1024 tokens for a longer, more coherent piece.

### Sweep Generation Parameters

```bash
cd src
python paramsearch.py
```

Runs a grid over 12 temperature × top-k combinations (0.3/5 through 1.0/50) and exports each as a separate `.mid` file alongside the original reference song for easy comparison.

### Evaluate

```bash
cd src
python testLoss.py
```

Reports average cross-entropy loss over the full dataset using the best checkpoint.

---

## How It Works

### Training

- Each MIDI file is converted to an integer token sequence bookended by `<SOS>` and `<EOS>`.
- Sequences are sliced into **overlapping windows** of 512 tokens (stride = 256) to increase training samples.
- The model is trained with **next-token prediction** using cross-entropy loss (ignoring `<PAD>`).
- Mixed-precision training (AMP + GradScaler) is used for faster GPU throughput.

### Generation

The `creation` function performs autoregressive **top-k sampling with temperature**:

1. Feed seed tokens through the model.
2. Scale the final-position logits by temperature.
3. Zero out all logits below the top-k threshold.
4. Sample the next token from the resulting distribution.
5. Append and repeat until `<EOS>` or the maximum length.

The generated integer sequence is reverse-mapped through the vocabulary, decoded back to structured tokens (`P_V_D` / `REST_T`), and converted to a playable MIDI file via `mido`.

---

## License

No license specified. Contact the repository owner for usage terms.
