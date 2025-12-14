#!/usr/bin/env bash
# Test Q3_HIFI quantization format
# This test:
#   1. Uses a pre-quantized Q3_HIFI model
#   2. Runs perplexity test
#   3. Asserts PPL is reasonable (<25)
#
# Usage:
#   ./tests/test-q3-hifi.sh <model_path>
#
# Note: Q3_HIFI requires tensor dimensions divisible by 256.
#       Small models like stories15M (288 dims) are not compatible.

set -e

# Configuration
PPL_THRESHOLD=25.0
TEST_TEXT="tests/test-q3-hifi-text.txt"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <q3_hifi_model_path>"
    echo "Example: $0 models/Qwen3-1.7B-Q3_HIFI.gguf"
    exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "Testing Q3_HIFI model: $MODEL_PATH"

# Create test text file if not present
if [ ! -f "$TEST_TEXT" ]; then
    echo "Creating test text file..."
    cat > "$TEST_TEXT" << 'EOF'
Once upon a time, there was a little girl named Lily. She loved to play in the garden with her dog Max.
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
EOF
fi

# Run perplexity test
echo "Running perplexity test..."
PPL_OUTPUT=$(./llama-perplexity -m "$MODEL_PATH" -f "$TEST_TEXT" -c 256 --chunks 2 2>&1)

# Extract final perplexity value
# Format: "Final estimate: PPL = X.XXXX +/- Y.YYYY"
PPL=$(echo "$PPL_OUTPUT" | grep -oP "Final estimate: PPL = \K[0-9]+\.[0-9]+" || echo "")

if [ -z "$PPL" ]; then
    # Try alternate format: just look for the last PPL value
    PPL=$(echo "$PPL_OUTPUT" | grep -oP "PPL = \K[0-9]+\.[0-9]+" | tail -1 || echo "")
fi

if [ -z "$PPL" ]; then
    echo "Error: Could not extract perplexity from output"
    echo "Output was:"
    echo "$PPL_OUTPUT"
    exit 1
fi

echo "Perplexity: $PPL"
echo "Threshold: $PPL_THRESHOLD"

# Check if PPL is reasonable (less than threshold)
if (( $(echo "$PPL < $PPL_THRESHOLD" | bc -l) )); then
    echo "✅ Test PASSED: PPL ($PPL) is below threshold ($PPL_THRESHOLD)"
    exit 0
else
    echo "❌ Test FAILED: PPL ($PPL) exceeds threshold ($PPL_THRESHOLD)"
    exit 1
fi

