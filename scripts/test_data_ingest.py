#!/usr/bin/env python
"""Simple sanity test for scripts/data_ingest.py."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_ingest():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.data_ingest import ingest_corpus

    return ingest_corpus


def main() -> None:
    ingest_corpus = load_ingest()
    with (
        tempfile.TemporaryDirectory() as raw_dir_name,
        tempfile.TemporaryDirectory() as clean_dir_name,
    ):
        raw_dir = Path(raw_dir_name)
        clean_dir = Path(clean_dir_name)
        (raw_dir / "a.txt").write_text(
            "Hello WORLD  \n\nSecond line.", encoding="utf-8"
        )
        (raw_dir / "nested").mkdir(parents=True, exist_ok=True)
        (raw_dir / "nested" / "b.txt").write_text(
            "Third\tLINE\n\n fourth   line ", encoding="utf-8"
        )
        output_file = clean_dir / "corpus.txt"
        total = ingest_corpus(raw_dir, output_file, lowercase=True)
        contents = output_file.read_text(encoding="utf-8").strip().splitlines()

        if not output_file.exists() or not contents:
            raise SystemExit("Cleaned corpus was not created.")
        assert total == len(contents), "Line count mismatch"
        print("[test_data_ingest] OK — cleaned corpus lines:", contents)


if __name__ == "__main__":
    main()
