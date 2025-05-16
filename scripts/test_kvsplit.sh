#!/bin/bash
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"
MODEL_PATH="${SCRIPT_DIR}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Run a simple test
cd "${LLAMA_CPP_DIR}" || exit 1

# Test with default settings
echo -e "\n=== Testing with default settings (FP16) ==="
./main -m "${MODEL_PATH}" -p "Hello, world" -n 16 -t 4 -fa 0 -t 8

# Test with K8V4 (8-bit keys, 4-bit values)
echo -e "\n\n=== Testing with K8V4 ==="
./main -m "${MODEL_PATH}" -p "Hello, world" -n 16 -t 4 -fa 0 -t 8 --kvq-key 8 --kvq-val 4
