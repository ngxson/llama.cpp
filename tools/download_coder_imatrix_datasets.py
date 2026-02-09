#!/usr/bin/env python3
"""
Create a high-fidelity mixed-domain dataset for HIFI imatrix generation,
optimized for code-aware quantization of LLMs like Qwen3.

Target Mix:
- 50% Clean Code (The Stack top langs + CodeSearchNet)
- 15% Code Instructions (CodeAlpaca / Evol-Instruct-Code style)
- 15% Technical Q&A (Stack Overflow + GitHub Issues)
- 10% Developer Docs (READMEs, API docs)
- 10% General Tech Knowledge (Wikipedia CS + ArXiv abstracts)

Usage:
    python create_hifi_imatrix_dataset.py --output hifi-imatrix-dataset.txt
"""

import argparse
import random
from typing import List, Optional, Dict, Any
from datasets import load_dataset

def read_or_generate(
    source: str,
    split: str = "train",
    text_key: str = "text",
    max_samples: int = 50000,
    min_length: int = 20,
    filter_fn=None
) -> List[str]:
    """Load or generate lines from a Hugging Face dataset."""
    print(f"Loading {source} ({split})...")
    try:
        ds = load_dataset(source, split=split, streaming=True)
    except Exception as e:
        print(f"⚠️ Failed to load {source}: {e}")
        return []

    lines = []
    for item in ds:
        if len(lines) >= max_samples:
            break
        text = item.get(text_key, "").strip()
        if not text:
            continue
        if len(text) < min_length:
            continue
        if filter_fn and not filter_fn(item):
            continue
        lines.append(text)
    print(f"  → Got {len(lines)} samples")
    return lines

def main():
    parser = argparse.ArgumentParser(description="Build HIFI imatrix dataset")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)

    # === 1. Clean Code Repositories (50%) ===
    code_lines = []

    # The Stack v2 - top languages only (Python, JS, TS, Java, C++, Go, Rust, C#, PHP, Ruby)
    stack_langs = ["Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "C#", "PHP", "Ruby"]
    for lang in stack_langs:
        lines = read_or_generate(
            "bigcode/the-stack-v2-dedup",
            split="train",
            text_key="content",
            max_samples=3000,  # ~30k total
            min_length=30,
            filter_fn=lambda x: x.get("lang") == lang and x.get("size") > 100
        )
        code_lines.extend(lines)

    # CodeSearchNet (high-quality GitHub snippets)
    codesearchnet = read_or_generate(
        "code_search_net",
        split="train",
        text_key="whole_func_string",
        max_samples=10000,
        min_length=50
    )
    code_lines.extend(codesearchnet)

    # === 2. Code Instructions (15%) ===
    instruct_lines = []

    # CodeAlpaca (instruction-response pairs)
    codealpaca = read_or_generate(
        "sahil2801/CodeAlpaca-20k",
        split="train",
        text_key="text",
        max_samples=5000,
        min_length=30
    )
    instruct_lines.extend(codealpaca)

    # Evol-Instruct-Code (synthetic but high-quality)
    evolinstruct = read_or_generate(
        "nickrosh/Evol-Instruct-Code-80k-v1",
        split="train",
        text_key="output",
        max_samples=5000,
        min_length=30
    )
    instruct_lines.extend(evolinstruct)

    # === 3. Technical Q&A (15%) ===
    qa_lines = []

    # Stack Overflow (questions + answers)
    so = read_or_generate(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        text_key="response",
        max_samples=7500,
        min_length=40
    )
    qa_lines.extend(so)

    # GitHub issues (filtered for technical discussions)
    gh_issues = read_or_generate(
        "m-a-p/CodeFeedback-Filtered",
        split="train",
        text_key="answer",
        max_samples=7500,
        min_length=40
    )
    qa_lines.extend(gh_issues)

    # === 4. Developer Docs (10%) ===
    doc_lines = []

    # GitHub READMEs from popular repos
    readmes = read_or_generate(
        "bigcode/stack-readmes",
        split="train",
        text_key="readme",
        max_samples=5000,
        min_length=50
    )
    doc_lines.extend(readmes)

    # API documentation snippets
    api_docs = read_or_generate(
        "nomic-ai/gpt4all-j-prompt-generations",
        split="train",
        text_key="prompt",
        max_samples=5000,
        min_length=30,
        filter_fn=lambda x: "api" in x.get("prompt", "").lower() or "function" in x.get("prompt", "").lower()
    )
    doc_lines.extend(api_docs)

    # === 5. General Tech Knowledge (10%) ===
    general_lines = []

    # Wikipedia (CS-related only)
    wiki_cs = read_or_generate(
        "wikipedia",
        split="train",
        text_key="text",
        max_samples=5000,
        min_length=60,
        filter_fn=lambda x: any(kw in x.get("title", "").lower() for kw in [
            "algorithm", "data structure", "computer science", "programming", "software",
            "machine learning", "artificial intelligence", "compiler", "operating system"
        ])
    )
    general_lines.extend(wiki_cs)

    # ArXiv CS abstracts
    arxiv = read_or_generate(
        "ccdv/arxiv-summarization",
        split="train",
        text_key="abstract",
        max_samples=5000,
        min_length=80
    )
    general_lines.extend(arxiv)

    # === Normalize counts based on target weights ===
    total_target = 100_000  # total lines desired
    targets = {
        'code': int(0.50 * total_target),
        'instruct': int(0.15 * total_target),
        'qa': int(0.15 * total_target),
        'docs': int(0.10 * total_target),
        'general': int(0.10 * total_target),
    }

    def truncate_or_sample(lst: List[str], n: int) -> List[str]:
        if len(lst) <= n:
            return lst
        return random.sample(lst, n)

    final_lines = []
    final_lines.extend(truncate_or_sample(code_lines, targets['code']))
    final_lines.extend(truncate_or_sample(instruct_lines, targets['instruct']))
    final_lines.extend(truncate_or_sample(qa_lines, targets['qa']))
    final_lines.extend(truncate_or_sample(doc_lines, targets['docs']))
    final_lines.extend(truncate_or_sample(general_lines, targets['general']))

    # Shuffle final dataset
    random.shuffle(final_lines)

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line.replace('\n', ' ') + '\n')

    print(f"\n✅ Created HIFI imatrix dataset: {args.output}")
    print(f"   Total lines: {len(final_lines)}")

if __name__ == "__main__":
    main()