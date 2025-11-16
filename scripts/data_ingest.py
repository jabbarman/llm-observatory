#!/usr/bin/env python
"""Load and clean raw text files into a single training corpus."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

WHITESPACE_RE = re.compile(r"\s+")


def list_text_files(input_dir: Path) -> List[Path]:
    """Collect all .txt files under the input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted(p for p in input_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    return files


def normalize_lines(text: str, lowercase: bool) -> Iterable[str]:
    """Yield cleaned lines from a text blob."""
    for raw_line in text.splitlines():
        normalized = raw_line.strip()
        if not normalized:
            continue
        normalized = WHITESPACE_RE.sub(" ", normalized)
        if lowercase:
            normalized = normalized.lower()
        if normalized:
            yield normalized


def ingest_corpus(
    input_dir: Path,
    output_file: Path,
    lowercase: bool = False,
) -> int:
    """Clean every .txt file under input_dir and write the merged corpus."""
    cleaned_lines: List[str] = []
    for path in list_text_files(input_dir):
        content = path.read_text(encoding="utf-8")
        cleaned_lines.extend(normalize_lines(content, lowercase))

    if not cleaned_lines:
        raise RuntimeError("Ingestion produced no non-empty lines.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
    return len(cleaned_lines)


def parse_args() -> argparse.Namespace:
    desc = "Clean raw text into a corpus."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_raw"),
        help="Directory containing raw .txt files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data_clean/corpus.txt"),
        help="Destination for the cleaned corpus.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase text before writing the corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_lines = ingest_corpus(
        input_dir=args.input_dir,
        output_file=args.output_file,
        lowercase=args.lowercase,
    )
    print(
        f"[data_ingest] Wrote {total_lines} lines to {args.output_file} "
        f"(lowercase={args.lowercase})"
    )


if __name__ == "__main__":
    main()
