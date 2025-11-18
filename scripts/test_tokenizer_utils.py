#!/usr/bin/env python
"""Test tokenizer utils batch encode and shard writing."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _import_utils():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tokenizer.train_tokenizer import train_sentencepiece
    from tokenizer.utils import SentencePieceBatchEncoder, save_token_shards

    return train_sentencepiece, SentencePieceBatchEncoder, save_token_shards


def main() -> None:
    corpus = "hello world\nAnother test line.\n"
    (
        train_sentencepiece,
        SentencePieceBatchEncoder,
        save_token_shards,
    ) = _import_utils()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        corpus_path = tmp_path / "corpus.txt"
        corpus_path.write_text(corpus, encoding="utf-8")
        model_prefix = tmp_path / "toy"
        train_sentencepiece(
            corpus_path=corpus_path,
            model_prefix=model_prefix,
            vocab_size=64,
            character_coverage=1.0,
        )
        encoder = SentencePieceBatchEncoder(model_prefix.with_suffix(".model"))
        encoded = encoder.encode_batch(["Hello world", "test line"])
        shard_dir = tmp_path / "shards"
        shards = save_token_shards(
            encoded,
            shard_dir,
            shard_size=1,
            metadata={"source": "unit-test"},
        )
        assert len(shards) == 2
        loaded = np.load(shards[0], allow_pickle=True)["tokens"]
        if loaded.size == 0:
            raise SystemExit("Shard file is empty.")
        print("[test_tokenizer_utils] shards:", shards)


if __name__ == "__main__":
    main()
