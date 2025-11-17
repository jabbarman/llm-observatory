#!/usr/bin/env python
"""Train a SentencePiece tokenizer on the cleaned corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def train_sentencepiece(
    corpus_path: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
) -> None:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data_clean/corpus.txt"),
        help="Cleaned corpus path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenizer"),
        help="Directory to store tokenizer artifacts.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size for the tokenizer.",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage fraction for SentencePiece.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.output_dir / "spm"
    train_sentencepiece(
        corpus_path=args.corpus,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )
    print(
        f"[train_tokenizer] Saved tokenizer to {prefix.with_suffix('.model')} "
        f"and {prefix.with_suffix('.vocab')}"
    )


if __name__ == "__main__":
    main()
