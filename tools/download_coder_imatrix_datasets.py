#!/usr/bin/env python3
"""
Fast imatrix dataset builder for Qwen3-coder.
Uses lightweight, pre-filtered sources to avoid hangs on big datasets.
Target: >5000 high-quality samples in <10 mins.
"""

import argparse
import random
from typing import List
from datasets import load_dataset

def safe_load(
    name: str,
    split: str = "train",
    text_key: str = "text",
    max_samples: int = 5000,
    min_chars: int = 30
) -> List[str]:
    """Safely load dataset with timeout-friendly streaming."""
    print(f"Loading {name}...")
    try:
        ds = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Skipping {name}: {e}")
        return []

    lines = []
    for item in ds:
        if len(lines) >= max_samples:
            break
        text = item.get(text_key, "").strip()
        if len(text) >= min_chars:
            lines.append(text)
    print(f"  → {len(lines)} samples")
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    all_lines = []

    # === 1. Code Repositories (50%) ===
    # Use The Stack v1.2 (filtered) - FAST and permissive
    stack = safe_load(
        "bigcode/the-stack-dedup",
        split="train",
        text_key="content",
        max_samples=25000,
        min_chars=40
    )
    all_lines.extend(stack)

    # CodeSearchNet (fast, high-quality)
    codesearchnet = safe_load(
        "code_search_net",
        split="train",
        text_key="whole_func_string",
        max_samples=10000,
        min_chars=50
    )
    all_lines.extend(codesearchnet)

    # === 2. Code Instructions (15%) ===
    codealpaca = safe_load(
        "sahil2801/CodeAlpaca-20k",
        max_samples=7500,
        min_chars=30
    )
    all_lines.extend(codealpaca)

    # === 3. Technical Q&A (15%) ===
    stackoverflow = safe_load(
        "HuggingFaceH4/stack-exchange-preferences",
        text_key="response",
        max_samples=7500,
        min_chars=40
    )
    all_lines.extend(stackoverflow)

    # === 4. Developer Docs (10%) ===
    readmes = safe_load(
        "bigcode/stack-readmes",
        text_key="readme",
        max_samples=5000,
        min_chars=50
    )
    all_lines.extend(readmes)

    # === 5. General Tech (10%) ===
    wiki = safe_load(
        "wikitext",
        "wikitext-103-raw-v1",
        max_samples=5000,
        min_chars=60
    )
    all_lines.extend(wiki)

    # Shuffle and cap at 50k+ lines
    random.shuffle(all_lines)
    final_lines = all_lines[:50000]

    # Write
    with open(args.output, 'w') as f:
        for line in final_lines:
            f.write(line.replace('\n', ' ') + '\n')

    print(f"\n✅ Done! {len(final_lines)} lines saved to {args.output}")

if __name__ == "__main__":
    main()