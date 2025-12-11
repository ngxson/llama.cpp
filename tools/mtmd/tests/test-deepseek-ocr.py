#!/usr/bin/env python3
"""
Test script to compare llama.cpp mtmd-cli output with HuggingFace reference implementation
for DeepSeek-OCR model using embedding similarity.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers import util


def run_mtmd_deepseek_ocr(
        model_path: str,
        mmproj_path: str,
        image_path: str,
        bin_path: str
) -> str:
    """
    Run inference using llama.cpp mtmd-cli.
    """
    cmd = [
        bin_path,
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--image", image_path,
        # "-p", "<|grounding|>Convert the document to markdown.",
        "-p", "Free OCR.",
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "-n", "1024",
        # "--verbose"
    ]

    print(f"Running llama.cpp command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        timeout=300
    )

    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='replace')
        print(f"llama.cpp stderr: {stderr}")
        raise RuntimeError(f"llama-mtmd-cli failed with code {result.returncode}")

    output = result.stdout.decode('utf-8', errors='replace').strip()
    print(f"llama.cpp output length: {len(output)} chars")
    return output


def run_mtmd_qwen_vl(
        model_path: str,
        mmproj_path: str,
        image_path: str,
        prompt: str,
        bin_path: str
) -> str:
    """
    Run inference using llama.cpp mtmd-cli with Qwen2.5-VL model.
    """
    cmd = [
        bin_path,
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--image", image_path,
        "-p", prompt,
        "--temp", "0"
    ]

    print(f"Running llama.cpp command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        print(f"llama.cpp stderr: {result.stderr}")
        raise RuntimeError(f"llama-mtmd-cli failed with code {result.returncode}")

    output = result.stdout.strip()
    print(f"llama.cpp output length: {len(output)} chars")
    return output


def compute_embedding_similarity(text1: str, text2: str, model_name: str) -> float:
    """
    Compute cosine similarity between two texts using embedding model.
    """
    print(f"Loading embedding model: {model_name}")

    # Use sentence-transformers for easier embedding extraction
    # For Gemma embedding, we use the sentence-transformers wrapper
    try:
        embed_model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception:
        # Fallback to a commonly available model if Gemma embedding not available
        print(f"Could not load {model_name}, falling back to all-MiniLM-L6-v2")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Computing embeddings...")
    embeddings = embed_model.encode([text1, text2], convert_to_numpy=True)

    similarity = util.similarity.cos_sim([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)


def main():
    ap = argparse.ArgumentParser(description="Compare llama.cpp and HuggingFace DeepSeek-OCR outputs")
    ap.add_argument("--hf-model", default="Dogacel/DeepSeek-OCR-Metal-MPS",
                    help="HuggingFace model ID")
    ap.add_argument("--llama-model", default="gguf_models/deepseek-ai/deepseek-ocr-f16.gguf",
                    help="Path to llama.cpp GGUF model")
    ap.add_argument("--mmproj", default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-f16.gguf",
                    help="Path to mmproj GGUF file")
    ap.add_argument("--image", default="test-1.jpeg",
                    help="Path to test image")
    ap.add_argument("--llama-bin", default="build/bin/llama-mtmd-cli",
                    help="Path to llama-mtmd-cli binary")
    ap.add_argument("--embedding-model", default="google/embeddinggemma-300m",
                    help="Embedding model for similarity computation")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Minimum similarity threshold for pass")
    args = ap.parse_args()

    # Validate paths
    # script directory + image
    mtmd_dir = Path(__file__).parent.parent
    args.image = str(mtmd_dir / args.image)
    # project directory + llama model
    args.llama_model = str(mtmd_dir.parent.parent / args.llama_model)
    # project directory + mmproj
    args.mmproj = str(mtmd_dir.parent.parent / args.mmproj)
    args.llama_bin = str(mtmd_dir.parent.parent / args.llama_bin)
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    if not Path(args.llama_model).exists():
        print(f"Error: Model not found: {args.llama_model}")
        sys.exit(1)
    if not Path(args.mmproj).exists():
        print(f"Error: mmproj not found: {args.mmproj}")
        sys.exit(1)

    print("=" * 60)
    print("DeepSeek-OCR: llama.cpp vs HuggingFace Comparison")
    print("=" * 60)

    # Default paths based on your command

    qwen_vl_out = run_mtmd_qwen_vl(
        model_path=str(mtmd_dir.parent.parent / "gguf_models/qwen/Qwen2.5-VL-7B-Instruct-f16.gguf"),
        mmproj_path=str(mtmd_dir.parent.parent / "gguf_models/qwen/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"),
        image_path=args.image,
        prompt="tell me what do you see in this picture?",
        bin_path=args.llama_bin
    )

    # Run llama.cpp inference
    print("\n[2/3] Running llama.cpp implementation...")
    llama_output = run_mtmd_deepseek_ocr(
        args.llama_model,
        args.mmproj,
        args.image,
        args.llama_bin
    )

    # Compute similarity
    print("\n[3/3] Computing embedding similarity...")
    similarity = compute_embedding_similarity(
        qwen_vl_out,
        llama_output,
        args.embedding_model
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nQwen2.5-VL output:\n{'-' * 40}")
    print(qwen_vl_out)
    print(f"\nDeepSeek-OCR output:\n{'-' * 40}")
    print(llama_output)
    print(f"\n{'=' * 60}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Result: {'PASS' if similarity >= args.threshold else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if similarity >= args.threshold else 1)


if __name__ == "__main__":
    main()
