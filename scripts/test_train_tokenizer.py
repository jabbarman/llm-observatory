#!/usr/bin/env python
"""Smoke test for tokenizer training + round trip."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tokenizer.train_tokenizer import train_sentencepiece

    return train_sentencepiece


def main() -> None:
    train_sentencepiece = load_trainer()
    corpus = "hello world\nThis is A Test.\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text(corpus, encoding="utf-8")
        prefix = tmp_path / "toy"

        train_sentencepiece(
            corpus_path=corpus_path,
            model_prefix=prefix,
            vocab_size=128,
            character_coverage=1.0,
        )
        sp = spm.SentencePieceProcessor()
        sp.load(str(prefix.with_suffix(".model")))
        sample = "Hello world!"
        ids = sp.encode(sample, out_type=int)
        decoded = sp.decode(ids)
        if not decoded:
            raise SystemExit("Decoded string is empty.")
        print("[test_train_tokenizer] encode:", ids)
        print("[test_train_tokenizer] decode:", decoded)


if __name__ == "__main__":
    main()
