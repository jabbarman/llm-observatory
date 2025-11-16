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
