# Data Sources & Policy

## Rules
- ✅ Use only datasets with open, permissive licenses (Apache-2.0, MIT, CC-BY, CC-BY-SA, ODC-BY).
- ❌ Do not ingest personal data (PII), copyrighted books, or unverified scrapes.
- ✅ Record provenance for every dataset: name, version, license, and hash.
- ✅ Apply deduplication + filtering before tokenization.
- ✅ Keep raw → clean → tokenized pipeline reproducible.

## Provenance Log

| Dataset         | License   | Source URL                     | SHA256 Hash (raw) | Notes |
|-----------------|-----------|--------------------------------|------------------|-------|
| tiny-wiki-sample| CC-BY-SA  | https://dumps.wikimedia.org    | `<to-fill>`      | Used for tokenizer tests |
| <future-dataset>| <license> | <url>                          | `<to-fill>`      |       |
