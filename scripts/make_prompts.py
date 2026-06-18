#!/usr/bin/env python3
"""Generate local prompt files for the mlxfast harness.

Creates:
  prompts/benchmark_local.txt  — BENCHMARK_PROMPT_TOKENS tokens of natural language

The correctness prompt (prompts/correctness_local.txt) is short enough
to be written by hand and is already committed to the repo.

Source text: Project Gutenberg public-domain books concatenated until the
target token count is reached. The tokenizer used is the one bundled with
the reference model weights (must be downloaded first via `mlxfast weights`).

Usage:
  python scripts/make_prompts.py

Re-run any time the tokenizer or target token count changes.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Public-domain sources from Project Gutenberg (plain text, UTF-8).
# These are large enough that a single book covers 32k tokens.
# ---------------------------------------------------------------------------
SOURCES = [
    # Moby Dick — Herman Melville (~215k words)
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    # Pride and Prejudice — Jane Austen (~122k words, fallback)
    "https://www.gutenberg.org/files/1342/1342-0.txt",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUT_FILE = PROMPTS_DIR / "benchmark_local.txt"

# Import harness constants for the target token count.
sys.path.insert(0, str(REPO_ROOT))
from mlxfast.harness.constants import (
    BENCHMARK_PROMPT_TOKENS,
    REFERENCE_WEIGHTS_DIR,
    REFERENCE_MODEL_DIRNAME,
)


def _fetch_text(url: str) -> str:
    print(f"  Fetching {url} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    print(f"{len(raw):,} chars")
    return raw


def _load_tokenizer():
    tokenizer_path = REFERENCE_WEIGHTS_DIR / REFERENCE_MODEL_DIRNAME
    if not tokenizer_path.exists():
        sys.exit(
            f"Reference weights not found at {tokenizer_path}.\n"
            "Run `mlxfast weights` first, then re-run this script."
        )
    try:
        import mlx_lm
        _, tokenizer = mlx_lm.load(str(tokenizer_path))
        return tokenizer
    except Exception as e:
        sys.exit(f"Failed to load tokenizer from {tokenizer_path}: {e}")


def main() -> None:
    print(f"Target: {BENCHMARK_PROMPT_TOKENS:,} tokens → {OUTPUT_FILE}")

    print("Loading tokenizer...")
    tokenizer = _load_tokenizer()

    # Fetch source texts until we have enough tokens.
    combined = ""
    for url in SOURCES:
        combined += _fetch_text(url) + "\n\n"
        ids = tokenizer.encode(combined)
        print(f"  Tokenised so far: {len(ids):,} tokens")
        if len(ids) >= BENCHMARK_PROMPT_TOKENS:
            break
    else:
        ids = tokenizer.encode(combined)
        if len(ids) < BENCHMARK_PROMPT_TOKENS:
            sys.exit(
                f"Not enough tokens after fetching all sources: "
                f"{len(ids):,} < {BENCHMARK_PROMPT_TOKENS:,}. "
                "Add more sources to SOURCES."
            )

    # Decode exactly BENCHMARK_PROMPT_TOKENS tokens back to text so the
    # file contains clean UTF-8 prose (not a raw token ID list).
    truncated_ids = ids[:BENCHMARK_PROMPT_TOKENS]
    text = tokenizer.decode(truncated_ids)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(f"Written {len(text):,} chars ({BENCHMARK_PROMPT_TOKENS:,} tokens) → {OUTPUT_FILE}")

    # Verify round-trip.
    verify_ids = tokenizer.encode(text)
    if len(verify_ids) < BENCHMARK_PROMPT_TOKENS:
        print(
            f"Warning: round-trip tokenisation gives {len(verify_ids):,} tokens "
            f"(expected >= {BENCHMARK_PROMPT_TOKENS:,}). "
            "The harness _tokenize() will raise at run time. "
            "Try increasing the source text length slightly."
        )
    else:
        print("Round-trip check passed.")


if __name__ == "__main__":
    main()
