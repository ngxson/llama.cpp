#!/bin/bash

# make sure we are in the right directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

#export LLAMA_CACHE="$SCRIPT_DIR/tmp"

set -eu

PROJ_ROOT="$SCRIPT_DIR/../.."
cd $PROJ_ROOT

###############

arr_bin=()
arr_hf=()

add_test() {
    local bin=$1
    local hf=$2
    arr_bin+=("$bin")
    arr_hf+=("$hf")
}

add_test "llama-gemma3-cli"   "ggml-org/gemma-3-4b-it-GGUF"
add_test "llama-llava-cli"    "guinmoon/MobileVLM-3B-GGUF"
add_test "llama-llava-cli"    "THUDM/glm-edge-v-5b-gguf"
add_test "llama-llava-cli"    "second-state/Llava-v1.5-7B-GGUF:Q2_K"
add_test "llama-llava-cli"    "cjpais/llava-1.6-mistral-7b-gguf:Q3_K"
add_test "llama-minicpmv-cli" "openbmb/MiniCPM-Llama3-V-2_5-gguf:Q2_K"
add_test "llama-minicpmv-cli" "openbmb/MiniCPM-V-2_6-gguf:Q2_K"
add_test "llama-qwen2vl-cli"  "bartowski/Qwen2-VL-2B-Instruct-GGUF"

###############

cmake --build build -j --target "${arr_bin[@]}"

for i in "${!arr_bin[@]}"; do
    bin="${arr_bin[$i]}"
    hf="${arr_hf[$i]}"

    echo "Running test with binary: $bin and HF model: $hf"
    echo ""
    echo ""

    "$PROJ_ROOT/build/bin/$bin" -hf "$hf" --image $PROJ_ROOT/media/llama1-logo.png -p "what do you see"

    echo ""
    echo ""
    echo ""
    echo "#################################################"
    echo "#################################################"
    echo ""
    echo ""
done
