"""
TAC-SM Data Preparation Script

Converts raw source code / bug report corpora into .pt token files
for training.

Usage:
  python scripts/prepare_data.py --input_dir ./raw --output_dir ./data/repair_corpus
  python scripts/prepare_data.py --synthetic --n 50000 --output_dir ./data/synthetic

Supported input formats:
  .py   — Python source files
  .json — Bug report JSON (expected fields: description, code, error, family)
  .txt  — Plain text

Each output .pt file contains a (T,) LongTensor of token ids.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Simple character-level tokenizer as fallback when no HF tokenizer is available
CHAR_VOCAB_SIZE = 256


class CharTokenizer:
    """Character-level tokenizer. vocab_size = 256."""

    def encode(self, text: str, max_length: int = 4096) -> List[int]:
        return [min(ord(c), 255) for c in text[:max_length]]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) for i in ids)


def get_tokenizer(vocab_size: int = 32000):
    """
    Return a tokenizer. Uses HuggingFace if available, else char-level.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("microsoft/phi-1_5", use_fast=True)
        print("  Using HuggingFace tokenizer (phi-1.5 BPE, vocab=50256)")
        return tok, tok.vocab_size
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", use_fast=True)
        print("  Using CodeLlama tokenizer (vocab=32000)")
        return tok, 32000
    except Exception:
        pass

    print("  Using character-level tokenizer (vocab=256). Install `transformers` for BPE.")
    tok = CharTokenizer()
    return tok, CHAR_VOCAB_SIZE


def encode(tokenizer, text: str, max_length: int) -> List[int]:
    if isinstance(tokenizer, CharTokenizer):
        return tokenizer.encode(text, max_length)
    return tokenizer.encode(text, max_length=max_length, truncation=True)


def process_python_file(path: Path, tokenizer, max_length: int) -> Optional[torch.Tensor]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        ids  = encode(tokenizer, text, max_length)
        if len(ids) < 32:
            return None
        return torch.tensor(ids, dtype=torch.long)
    except Exception as e:
        print(f"    Skip {path}: {e}")
        return None


def process_bug_json(path: Path, tokenizer, max_length: int) -> Optional[torch.Tensor]:
    try:
        data = json.loads(path.read_text())
        parts = []
        if "description" in data:
            parts.append(f"BUG: {data['description']}")
        if "error" in data:
            parts.append(f"ERROR: {data['error']}")
        if "code" in data:
            parts.append(f"CODE:\n{data['code']}")
        text = "\n".join(parts)
        ids  = encode(tokenizer, text, max_length)
        if len(ids) < 32:
            return None
        return torch.tensor(ids, dtype=torch.long)
    except Exception as e:
        print(f"    Skip {path}: {e}")
        return None


def make_synthetic(
    n: int,
    seq_len: int,
    vocab_size: int,
    output_dir: Path,
):
    """Generate n synthetic random token sequences for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} synthetic sequences (seq_len={seq_len}, vocab={vocab_size})...")
    for i in range(n):
        ids = torch.randint(1, vocab_size, (seq_len,))
        torch.save(ids, output_dir / f"syn_{i:07d}.pt")
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n}")
    print(f"Done. Saved to {output_dir}")


def process_directory(
    input_dir:  Path,
    output_dir: Path,
    tokenizer,
    max_length: int,
    max_files:  int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".py", ".json", ".txt", ".js", ".ts", ".go", ".java", ".cpp", ".c"}
    files      = sorted(f for f in input_dir.rglob("*") if f.suffix in extensions)

    if max_files > 0:
        files = files[:max_files]

    print(f"Processing {len(files)} files from {input_dir}...")
    saved = 0
    for i, path in enumerate(files):
        if path.suffix == ".json":
            tensor = process_bug_json(path, tokenizer, max_length)
        else:
            tensor = process_python_file(path, tokenizer, max_length)

        if tensor is not None:
            out_path = output_dir / f"{saved:07d}.pt"
            torch.save(tensor, out_path)
            saved += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(files)}, saved {saved}")

    print(f"Done. Saved {saved} token files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TAC-SM Data Preparation")
    parser.add_argument("--input_dir",  type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--synthetic",  action="store_true",
                        help="Generate synthetic random data")
    parser.add_argument("--n",          type=int, default=10000,
                        help="Number of synthetic sequences")
    parser.add_argument("--seq_len",    type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--max_files",  type=int, default=0,
                        help="Max files to process (0=all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.synthetic:
        make_synthetic(args.n, args.seq_len, args.vocab_size, output_dir)
        return

    if not args.input_dir:
        print("Error: --input_dir is required unless --synthetic is set")
        sys.exit(1)

    tokenizer, vocab_size = get_tokenizer(args.vocab_size)
    process_directory(
        input_dir  = Path(args.input_dir),
        output_dir = output_dir,
        tokenizer  = tokenizer,
        max_length = args.seq_len,
        max_files  = args.max_files,
    )


if __name__ == "__main__":
    main()
