#!/usr/bin/env python3
"""Download datasets for imatrix generation."""

from typing import Any, cast

from datasets import load_dataset

SAMPLE_SEPARATOR = "<|endofsample|>"


def download_mathqa(output_file="mathqa.txt", num_samples=10000) -> tuple[str, int, bool]:
    """Download MathQA problems. Returns (filename, expected_count, uses_separator)."""
    print(f"Downloading MathQA dataset ({num_samples} samples)...")
    ds = load_dataset('allenai/math_qa', revision='refs/convert/parquet', split='train')
    with open(output_file, 'w') as f:
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            f.write(item['Problem'].strip() + '\n')
    print(f"  Saved to {output_file}")
    return output_file, num_samples, False


def download_codeparrot(output_file="codeparrot.txt", num_samples=10000) -> tuple[str, int, bool]:
    """Download CodeParrot code snippets. Returns (filename, expected_count, uses_separator)."""
    print(f"Downloading CodeParrot dataset ({num_samples} samples)...")
    ds = load_dataset('codeparrot/codeparrot-valid-v2-near-dedup', split='train', streaming=True)
    with open(output_file, 'w') as f:
        count = 0
        for item in ds:
            if count >= num_samples:
                break
            code = cast(dict[str, Any], item)['content'].strip()
            if code and len(code) > 20:  # skip tiny snippets
                f.write(code + '\n' + SAMPLE_SEPARATOR + '\n')
                count += 1
    print(f"  Saved to {output_file}")
    return output_file, num_samples, True


def download_wikitext(output_file="wikitext.txt", num_lines=20000) -> tuple[str, int, bool]:
    """Download WikiText samples. Returns (filename, expected_count, uses_separator)."""
    print(f"Downloading WikiText dataset ({num_lines} lines)...")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    count = 0
    with open(output_file, 'w') as f:
        for item in ds:
            if count >= num_lines:
                break
            line = cast(dict[str, Any], item)['text']
            if line.strip():
                f.write(line.strip() + '\n')
                count += 1
    print(f"  Saved to {output_file}")
    return output_file, num_lines, False


def verify_file(filename: str, expected: int, uses_separator: bool) -> bool:
    """Verify that a file has the expected number of samples."""
    with open(filename, 'r') as f:
        content = f.read()
    if uses_separator:
        actual = content.count(SAMPLE_SEPARATOR)
        unit = "samples"
    else:
        actual = content.count('\n')
        unit = "lines"
    if actual == expected:
        print(f"  ✓ {filename}: {actual} {unit}")
        return True
    else:
        print(f"  ✗ {filename}: expected {expected}, got {actual} {unit}")
        return False


if __name__ == "__main__":
    results = [
        download_mathqa(),
        download_codeparrot(),
        download_wikitext(),
    ]

    print("\nVerifying downloads...")
    all_ok = all(verify_file(f, n, sep) for f, n, sep in results)

    if all_ok:
        print("\nDone! All files verified.")
    else:
        print("\nWarning: Some files have unexpected line counts.")
        exit(1)
