#!/usr/bin/env python3
"""
Test Q3_HIFI quantization format.

This test:
  1. Uses a pre-quantized Q3_HIFI model (or quantizes a compatible model)
  2. Runs perplexity test
  3. Asserts PPL is reasonable (<25)

Usage:
    python tests/test-q3-hifi.py [--build-dir BUILD_DIR] [--model MODEL_PATH]

Note: Q3_HIFI requires tensor dimensions divisible by 256. 
      Small models like stories15M (288 dims) are not compatible.
      Use a model with compatible dimensions (e.g., Qwen, Llama, Mistral).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Configuration  
PPL_THRESHOLD = 25.0  # Reasonable threshold for 3-bit quantization

# Need enough text to generate 1024+ tokens for perplexity test
TEST_TEXT = """Once upon a time, there was a little girl named Lily. She loved to play in the garden with her dog Max.
One sunny day, Lily found a shiny red ball under a big tree. She was so happy! She threw the ball for Max to catch.
Max ran very fast and caught the ball in his mouth. Lily clapped her hands and laughed. They played all afternoon.
When the sun started to set, Lily's mom called them inside for dinner. Lily gave Max a big hug and said goodnight.
The next morning, Lily woke up early. She looked out the window and saw it was raining. She felt sad because she could not play outside.
But then Max came to her room with a toy in his mouth. Lily smiled and played with Max inside the house.

The story of quantum computing begins in the early 1980s when physicist Richard Feynman proposed that quantum mechanical 
phenomena could be simulated more efficiently using a quantum computer than a classical one. This idea laid the foundation 
for what would become one of the most transformative technologies of the 21st century. Quantum computers leverage the 
principles of quantum mechanics, particularly superposition and entanglement, to perform computations that would be 
practically impossible for classical computers.

In a classical computer, information is processed using bits that can be either 0 or 1. However, quantum computers use 
quantum bits, or qubits, which can exist in a superposition of both 0 and 1 simultaneously. This property allows quantum 
computers to explore many possible solutions at once, potentially solving certain problems exponentially faster than 
classical computers. Entanglement, another quantum phenomenon, allows qubits to be correlated in ways that have no 
classical counterpart, enabling even more powerful computational capabilities.

The development of practical quantum computers has been a challenging endeavor. Qubits are extremely fragile and can 
lose their quantum properties through a process called decoherence when they interact with their environment. This has 
led researchers to explore various physical implementations of qubits, including superconducting circuits, trapped ions, 
topological qubits, and photonic systems. Each approach has its own advantages and challenges.

Major technology companies and research institutions around the world are racing to build more powerful and reliable 
quantum computers. IBM, Google, Microsoft, and several startups have made significant progress in recent years. In 2019, 
Google announced quantum supremacy, claiming their quantum computer performed a calculation that would take the world's 
most powerful classical supercomputer thousands of years. While the significance of this achievement was debated, it 
marked an important milestone in the field.

The potential applications of quantum computing are vast. In cryptography, quantum computers could break many of the 
encryption methods that currently protect our digital communications, while also enabling new forms of quantum encryption 
that are theoretically unbreakable. In drug discovery and materials science, quantum simulations could help design new 
molecules and materials with specific properties. Optimization problems in logistics, finance, and machine learning 
could also benefit from quantum speedups.

However, significant challenges remain before quantum computers become practically useful for most applications. Current 
quantum computers have limited numbers of qubits and high error rates. Researchers are working on quantum error correction 
techniques and building more reliable hardware. The field of quantum software is also developing, with new algorithms and 
programming frameworks being created to make quantum computing more accessible.

The intersection of quantum computing and artificial intelligence is particularly exciting. Quantum machine learning 
algorithms could potentially train models faster or find patterns in data that classical algorithms miss. Some researchers 
believe that quantum computers might eventually lead to more powerful forms of artificial intelligence, though this remains 
speculative. What is clear is that the development of quantum computing represents a fundamental shift in our computational 
capabilities that could have profound implications for science, technology, and society.
"""


def find_executable(name: str, build_dir: Path) -> Path:
    """Find an executable in the build directory."""
    # Check common locations
    candidates = [
        build_dir / "bin" / name,
        build_dir / "bin" / "Release" / name,
        build_dir / "bin" / "Debug" / name,
        build_dir / name,
    ]
    
    # Add .exe suffix on Windows
    if sys.platform == "win32":
        candidates = [Path(str(c) + ".exe") for c in candidates] + candidates
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(f"Could not find {name} in {build_dir}")


def run_command(cmd: list, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
    )
    return result


def extract_ppl(output: str) -> float:
    """Extract perplexity value from llama-perplexity output."""
    # Try "Final estimate: PPL = X.XXXX"
    match = re.search(r"Final estimate: PPL = ([0-9]+\.[0-9]+)", output)
    if match:
        return float(match.group(1))
    
    # Try just "PPL = X.XXXX" (last occurrence)
    matches = re.findall(r"PPL = ([0-9]+\.[0-9]+)", output)
    if matches:
        return float(matches[-1])
    
    raise ValueError(f"Could not extract PPL from output:\n{output}")


def main():
    parser = argparse.ArgumentParser(description="Test Q3_HIFI quantization")
    parser.add_argument("--build-dir", type=Path, default=Path("build"),
                        help="Build directory containing llama binaries")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to a Q3_HIFI quantized model (must have dims divisible by 256)")
    parser.add_argument("--threshold", type=float, default=PPL_THRESHOLD,
                        help=f"Maximum acceptable perplexity (default: {PPL_THRESHOLD})")
    args = parser.parse_args()
    
    build_dir = args.build_dir.resolve()
    model_path = args.model.resolve()
    threshold = args.threshold
    
    # Find executable
    try:
        perplexity_exe = find_executable("llama-perplexity", build_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've built llama.cpp first.")
        return 1
    
    print(f"Using perplexity: {perplexity_exe}")
    print(f"Testing model: {model_path}")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1
    
    print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.2f} MiB")
    
    # Create test text file
    test_text_path = Path("tests") / "test-q3-hifi-text.txt"
    test_text_path.parent.mkdir(parents=True, exist_ok=True)
    test_text_path.write_text(TEST_TEXT)
    
    # Run perplexity test with small context
    print("\n=== Running perplexity test ===")
    result = run_command([
        str(perplexity_exe),
        "-m", str(model_path),
        "-f", str(test_text_path),
        "-c", "256",  # Small context to reduce compute
        "--chunks", "2"  # Just 2 chunks for quick test
    ])
    
    output = result.stdout + result.stderr
    
    if result.returncode != 0:
        print(f"Perplexity test failed:\n{output}")
        return 1
    
    # Extract and check PPL
    try:
        ppl = extract_ppl(output)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"\nPerplexity: {ppl:.4f}")
    print(f"Threshold: {threshold}")
    
    if ppl < threshold:
        print(f"\n✅ Test PASSED: PPL ({ppl:.4f}) is below threshold ({threshold})", flush=True)
        return 0
    else:
        print(f"\n❌ Test FAILED: PPL ({ppl:.4f}) exceeds threshold ({threshold})", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

