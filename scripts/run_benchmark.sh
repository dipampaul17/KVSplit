#!/bin/bash
set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${SCRIPT_DIR}/llama.cpp"
MODEL_PATH="${SCRIPT_DIR}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Change to llama.cpp directory
cd "${LLAMA_CPP_DIR}" || exit 1

# Test configurations
CONFIGS=(
    "FP16"
    "K8V8"
    "K8V4"
    "K4V8"
    "K4V4"
)

# Sequence lengths to test
SEQUENCE_LENGTHS=(128 2048 4096 8192)

# Run benchmarks
for seq_len in "${SEQUENCE_LENGTHS[@]}"; do
    echo -e "\n=== Benchmarking sequence length: ${seq_len} ==="
    
    for config in "${CONFIGS[@]}"; do
        echo -e "\n--- Testing ${config} ---"
        
        # Set KV cache parameters based on config
        case $config in
            "FP16")
                KV_ARGS=""
                ;;
            "K8V8")
                KV_ARGS="--kvq-key 8 --kvq-val 8"
                ;;
            "K8V4")
                KV_ARGS="--kvq-key 8 --kvq-val 4"
                ;;
            "K4V8")
                KV_ARGS="--kvq-key 4 --kvq-val 8"
                ;;
            "K4V4")
                KV_ARGS="--kvq-key 4 --kvq-val 4"
                ;;
            *)
                echo "Unknown config: ${config}"
                continue
                ;;
        esac
        
        # Run the benchmark
        OUTPUT_FILE="${RESULTS_DIR}/benchmark_${config}_seq${seq_len}.txt"
        echo "Running: ./main -m ${MODEL_PATH} -p \"Benchmarking KV cache performance\" -n ${seq_len} -t 8 -fa 0 ${KV_ARGS}"
        
        # Run with timeout to prevent hanging
        timeout 5m ./main -m "${MODEL_PATH}" -p "Benchmarking KV cache performance" \
            -n ${seq_len} -t 8 -fa 0 ${KV_ARGS} 2>&1 | tee "${OUTPUT_FILE}" || \
            echo "Warning: Benchmark for ${config} with seq_len=${seq_len} timed out or failed"
    done
done

echo -e "\n=== Benchmarking complete! Results saved to ${RESULTS_DIR} ==="
