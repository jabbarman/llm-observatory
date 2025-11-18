# LLM Observatory

A hands-on lab to train, fine-tune, serve, and evaluate LLMs — from scratch to deployment.

## Goals
- Train a tiny 100–300M GPT from scratch
- Fine-tune a 7B with QLoRA
- Serve quantized models locally
- Add Retrieval-Augmented Generation (RAG)
- Evaluate and interpret results

## Getting Started
The project targets **Python 3.11**. Using another version (for example, 3.12)
can break the PyTorch + NumPy compatibility guarantees we rely on.

1. Create and activate the local virtual environment:

   ```bash
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   ```

2. Install the dependencies while respecting the pinned NumPy/PyTorch versions:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt -c constraints.txt
   ```

3. Validate the environment and logging stack:

   ```bash
   python scripts/env_check.py
   python tiny-train/log_demo.py
   ```

`tiny-train/log_demo.py` writes TensorBoard scalars into `./runs`, allowing you
to confirm that TensorBoard receives data before starting expensive training
jobs.

## Data Ingestion
Raw text dumps go inside `data_raw/` (each file should end in `.txt`). Once the
files are in place run:

```bash
python scripts/data_ingest.py --input-dir data_raw --output-file data_clean/corpus.txt --lowercase
```

This normalizes whitespace, drops empty lines, (optionally) lowercases content,
and writes the cleaned corpus to `data_clean/corpus.txt`. Run
`python scripts/test_data_ingest.py` to smoke-test the ingestion logic on a
small synthetic dataset.

## Tokenizer Training
With `data_clean/corpus.txt` ready, train a SentencePiece BPE tokenizer:

```bash
python tokenizer/train_tokenizer.py --vocab-size 32000 --character-coverage 0.9995
```

Artifacts land in `tokenizer/spm.model` and `tokenizer/spm.vocab`. Run the
smoke test via `python scripts/test_train_tokenizer.py` to verify the trained
tokenizer can encode/decode sample text.

## Tokenized Shards
After training the tokenizer, encode text datasets into chunked `.npz` shards:

```bash
python - <<'PY'
from tokenizer.utils import SentencePieceBatchEncoder, save_token_shards
from pathlib import Path

encoder = SentencePieceBatchEncoder(Path("tokenizer/spm.model"))
texts = ["example prompt", "another sample"]
encoded = encoder.encode_batch(texts)
save_token_shards(encoded, Path("data_clean/tokens"), shard_size=2)
PY
```

Use `python scripts/test_tokenizer_utils.py` for a self-contained smoke test.
