"""Utilities for batching text encoding and saving token shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import sentencepiece as spm


class SentencePieceBatchEncoder:
    """Wraps a SentencePiece model for batch encoding."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model missing: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[List[int]]:
        encoded: List[List[int]] = []
        for text in texts:
            ids = self.sp.encode(
                text,
                out_type=int,
                add_bos=add_bos,
                add_eos=add_eos,
            )
            encoded.append(ids)
        return encoded


def save_token_shards(
    encoded_batches: Iterable[Sequence[int]],
    output_dir: Path,
    shard_size: int,
    metadata: dict | None = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []
    current: List[Sequence[int]] = []
    shard_index = 0
    for sequence in encoded_batches:
        current.append(sequence)
        if len(current) >= shard_size:
            shard_paths.append(
                _write_shard(
                    current,
                    output_dir,
                    shard_index,
                    metadata,
                )
            )
            current = []
            shard_index += 1
    if current:
        shard_paths.append(
            _write_shard(
                current,
                output_dir,
                shard_index,
                metadata,
            )
        )
    return shard_paths


def _write_shard(
    sequences: List[Sequence[int]],
    output_dir: Path,
    shard_index: int,
    metadata: dict | None,
) -> Path:
    shard_path = output_dir / f"tokens_{shard_index:04d}.npz"
    arr = np.array(list(sequences), dtype=object)
    np.savez(shard_path, tokens=arr)
    if metadata:
        meta_path = shard_path.with_suffix(".json")
        meta_content = {"count": len(sequences), **metadata}
        meta_path.write_text(
            json.dumps(meta_content, indent=2),
            encoding="utf-8",
        )
    return shard_path
