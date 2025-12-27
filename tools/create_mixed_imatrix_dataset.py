#!/usr/bin/env python3
"""
Create an interleaved dataset file for mixed-domain imatrix generation.

Usage:
  python create_mixed_imatrix_dataset.py \
    --wikitext wikitext.txt \
    --code codeparrot.txt \
    --math mathqa.txt \
    --output mixed-imatrix_dataset.txt \
    --ratio 50,25,25
"""

import argparse
import random
from typing import List, Optional

def read_lines(filename: str, max_lines: Optional[int] = None) -> List[str]:
    """Read non-empty lines from file, optionally limiting count."""
    lines = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped:  # Skip empty lines
                lines.append(stripped)
                if max_lines and len(lines) >= max_lines:
                    break
    return lines

def interleave_datasets(
    wikitext: List[str],
    code: List[str],
    math: List[str],
    ratios: tuple = (50, 25, 25)
) -> List[str]:
    """Interleave datasets according to given ratios (percentages)."""
    wt_ratio, code_ratio, math_ratio = ratios
    total_ratio = wt_ratio + code_ratio + math_ratio
   
    # Normalize ratios to fractions
    wt_frac = wt_ratio / total_ratio
    code_frac = code_ratio / total_ratio
    math_frac = math_ratio / total_ratio
   
    # Calculate how many lines we can take from each (conservative estimate)
    min_multiplier = min(
        len(wikitext) / wt_frac if wt_frac > 0 else float('inf'),
        len(code) / code_frac if code_frac > 0 else float('inf'),
        len(math) / math_frac if math_frac > 0 else float('inf')
    )
   
    target_wt = int(min_multiplier * wt_frac)
    target_code = int(min_multiplier * code_frac)
    target_math = int(min_multiplier * math_frac)
   
    print(f"Using {target_wt} Wikitext, {target_code} Code, {target_math} Math lines")
   
    # Truncate to target counts
    wikitext = wikitext[:target_wt]
    code = code[:target_code]
    math = math[:target_math]
   
    # Create interleaved list
    mixed = []
    i = j = k = 0
   
    while i < len(wikitext) or j < len(code) or k < len(math):
        # Add Wikitext lines (highest ratio)
        for _ in range(2):  # 2x more frequent than others
            if i < len(wikitext):
                mixed.append(wikitext[i])
                i += 1
       
        # Add Code line
        if j < len(code):
            mixed.append(code[j])
            j += 1
       
        # Add Math line
        if k < len(math):
            mixed.append(math[k])
            k += 1
   
    return mixed

def main():
    parser = argparse.ArgumentParser(description="Create mixed imatrix dataset")
    parser.add_argument("--wikitext", required=True, help="Wikitext dataset file")
    parser.add_argument("--code", required=True, help="Code dataset file")
    parser.add_argument("--math", required=True, help="Math dataset file")
    parser.add_argument("--output", required=True, help="Output mixed dataset file")
    parser.add_argument("--ratio", default="50,25,25",
                        help="Ratios as WIKITEXT,CODE,MATH (default: 50,25,25)")
   
    args = parser.parse_args()
   
    # Parse ratios
    ratios = tuple(int(x) for x in args.ratio.split(','))
    if len(ratios) != 3:
        raise ValueError("Ratio must have exactly 3 values (e.g., 50,25,25)")
   
    # Load datasets
    print("Loading datasets...")
    wikitext_lines = read_lines(args.wikitext)
    code_lines = read_lines(args.code)
    math_lines = read_lines(args.math)
   
    print(f"Loaded {len(wikitext_lines)} Wikitext lines")
    print(f"Loaded {len(code_lines)} Code lines")
    print(f"Loaded {len(math_lines)} Math lines")
   
    # Interleave
    mixed_lines = interleave_datasets(wikitext_lines, code_lines, math_lines, ratios)
   
    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in mixed_lines:
            f.write(line + '\n')
   
    print(f"\n✅ Created mixed dataset: {args.output}")
    print(f"   Total lines: {len(mixed_lines)}")
   
    # Sample output
    print("\nFirst 10 lines:")
    for i, line in enumerate(mixed_lines[:10]):
        prefix = "WT" if i % 4 < 2 else "CD" if i % 4 == 2 else "MH"
        print(f"  {prefix}: {line[:60]}...")

if __name__ == "__main__":
    main()