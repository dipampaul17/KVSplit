#!/bin/bash

# Exit on error, undefined variables, and pipeline failures
set -euo pipefail

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display error messages
error_exit() {
    local line_number=$1
    local message=$2
    echo -e "${RED}Error on line ${line_number}: ${message}${NC}" >&2
    exit 1
}

# Trap errors
trap 'error_exit $LINENO "Command failed with status $?"' ERR

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}==> $1${NC}"
}

# Function to extract Metal allocation information
extract_metal_stats() {
    local logfile=$1
    local type=$2
    echo -e "${BLUE}Extracting Metal stats for $type:${NC}"
    
    # Check if log has Metal allocation information
    if grep -q "METAL ALLOC" "$logfile"; then
        echo -e "${GREEN}✓ Metal memory logging enabled${NC}"
        
        # Extract total allocated memory
        total_allocated=$(grep "METAL ALLOC" "$logfile" | grep -v "freed" | awk '{sum+=$3} END {print sum/1024/1024 " MB"}')
        echo "Total Metal memory allocated: $total_allocated"
        
        # Extract KV cache allocation if available
        kv_cache_alloc=$(grep -i "kv " "$logfile" | grep -i "cache" | grep "METAL ALLOC" | awk '{sum+=$3} END {if(sum>0) print sum/1024/1024 " MB"; else print "Not found"}')
        echo "KV cache memory: $kv_cache_alloc"
        
        # Extract K and V allocations separately if possible
        k_alloc=$(grep -i " k " "$logfile" | grep "METAL ALLOC" | awk '{sum+=$3} END {if(sum>0) print sum/1024/1024 " MB"; else print "Not found"}')
        v_alloc=$(grep -i " v " "$logfile" | grep "METAL ALLOC" | awk '{sum+=$3} END {if(sum>0) print sum/1024/1024 " MB"; else print "Not found"}')
        
        echo "K cache memory: $k_alloc"
        echo "V cache memory: $v_alloc"
    else
        echo -e "${RED}No Metal allocation information found. Try using -t 8 flag.${NC}"
    fi
    
    echo "----------------------------------------------"
}

# Function to verify KV cache type in logs
verify_kv_types() {
    local logfile=$1
    local expected_k_type=$2
    local expected_v_type=$3
    
    echo -e "${BLUE}Verifying KV cache types:${NC}"
    
    # Extract type info from logs
    if grep -q "type_k" "$logfile" && grep -q "type_v" "$logfile"; then
        k_type=$(grep "type_k" "$logfile" | head -1 | sed 's/.*type_k[^a-z0-9]*\([a-z0-9_]*\).*/\1/')
        v_type=$(grep "type_v" "$logfile" | head -1 | sed 's/.*type_v[^a-z0-9]*\([a-z0-9_]*\).*/\1/')
        
        echo "Found key type: $k_type (Expected: $expected_k_type)"
        echo "Found value type: $v_type (Expected: $expected_v_type)"
        
        # Basic verification
        if [[ "$k_type" == *"$expected_k_type"* ]]; then
            echo -e "${GREEN}✓ Key type matches expected type${NC}"
        else
            echo -e "${RED}× Key type does not match expected type${NC}"
        fi
        
        if [[ "$v_type" == *"$expected_v_type"* ]]; then
            echo -e "${GREEN}✓ Value type matches expected type${NC}"
        else
            echo -e "${RED}× Value type does not match expected type${NC}"
        fi
    else
        echo -e "${YELLOW}KV cache type information not found in logs${NC}"
    fi
    
    echo "----------------------------------------------"
}

# Main execution
print_section "KVSplit Validation Script"

# Get base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${BASE_DIR}/llama.cpp"
MODEL_PATH="${BASE_DIR}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
PATCH_PATH="${BASE_DIR}/patch/kvsplit_fixed.patch"
RESULTS_DIR="${BASE_DIR}/results"

# Ensure results directory exists
mkdir -p "${RESULTS_DIR}"

# Verify model exists
if [ ! -f "${MODEL_PATH}" ]; then
    error_exit "$LINENO" "Model not found at ${MODEL_PATH}. Run setup.sh first."
fi

# Check if patch exists
if [ ! -f "${PATCH_PATH}" ]; then
    error_exit "$LINENO" "Patch file not found at ${PATCH_PATH}"
fi

# Apply the patch
print_section "Applying KVSplit patch"
cd "${LLAMA_CPP_DIR}"
# Check if patch has already been applied
if grep -q "kvq-key" "${LLAMA_CPP_DIR}/common/arg.cpp"; then
    echo -e "${YELLOW}Patch appears to be already applied${NC}"
else
    # Apply the patch
    patch -p1 < "${PATCH_PATH}" || error_exit "$LINENO" "Failed to apply patch"
    echo -e "${GREEN}✓ Patch applied successfully${NC}"
fi

# Build llama.cpp with Metal support
print_section "Building llama.cpp with Metal support"
cd "${LLAMA_CPP_DIR}"

# Build with CMake instead of make, since that's how we set up in Step 1
echo "Building with CMake..."
rm -rf build 2>/dev/null || true
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_METAL=on \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="11.0" \
    -DLLAMA_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_METAL_SHADER_DEBUG=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DBUILD_SHARED_LIBS=OFF || error_exit "$LINENO" "CMake configuration failed"

# Build with multiple jobs
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu) || \
    error_exit "$LINENO" "Failed to build llama.cpp"

# Find the main executable
MAIN_EXEC=""
if [ -f "bin/main" ]; then
    MAIN_EXEC="bin/main"
elif [ -f "bin/llama-cli" ]; then
    MAIN_EXEC="bin/llama-cli"
else
    # Try to find any llama executable
    MAIN_EXEC=$(find . -name "llama-*" -type f -executable | grep -v '\.o$' | head -1)
    if [ -z "$MAIN_EXEC" ]; then
        error_exit "$LINENO" "Could not find main llama executable"
    fi
fi

echo -e "${GREEN}✓ Successfully built llama.cpp with Metal support${NC}"
echo "Main executable: ${MAIN_EXEC}"

cd "${LLAMA_CPP_DIR}"

# Run tests
print_section "Running validation tests"

# Test case 1: Baseline (FP16 - same precision for both)
echo -e "${BLUE}Test 1: Baseline (FP16)${NC}"
LOG_FILE="${RESULTS_DIR}/kvsplit_test_fp16.log"
echo "Running: ./${MAIN_EXEC} -m ${MODEL_PATH} -t 8 -n 32 -p \"Hello world\""
./${MAIN_EXEC} -m "${MODEL_PATH}" -t 8 -n 32 -p "Hello world" > "${LOG_FILE}" 2>&1 || {
    echo -e "${RED}Test 1 failed with exit code $?${NC}"
    error_exit "$LINENO" "Baseline test failed"
}
echo -e "${GREEN}✓ Test 1 completed successfully${NC}"
extract_metal_stats "${LOG_FILE}" "FP16"
verify_kv_types "${LOG_FILE}" "f16" "f16"

# Test case 2: Using existing --kvq parameter (Q8_0 - same precision for both)
echo -e "${BLUE}Test 2: Using existing --kvq parameter (Q8_0)${NC}"
LOG_FILE="${RESULTS_DIR}/kvsplit_test_kvq_q8.log"
echo "Running: ./${MAIN_EXEC} -m ${MODEL_PATH} --kvq 8 -t 8 -n 32 -p \"Hello world\""
./${MAIN_EXEC} -m "${MODEL_PATH}" --kvq 8 -t 8 -n 32 -p "Hello world" > "${LOG_FILE}" 2>&1 || {
    echo -e "${RED}Test 2 failed with exit code $?${NC}"
    error_exit "$LINENO" "Test using --kvq parameter failed"
}
echo -e "${GREEN}✓ Test 2 completed successfully${NC}"
extract_metal_stats "${LOG_FILE}" "Q8_0"
verify_kv_types "${LOG_FILE}" "q8_0" "q8_0"

# Test case 3: Split precision (K8V4 - 8-bit keys, 4-bit values)
echo -e "${BLUE}Test 3: Split precision (K8V4 - 8-bit keys, 4-bit values)${NC}"
LOG_FILE="${RESULTS_DIR}/kvsplit_test_k8v4.log"
echo "Running: ./${MAIN_EXEC} -m ${MODEL_PATH} --kvq-key 8 --kvq-val 4 -t 8 -n 32 -p \"Hello world\""
./${MAIN_EXEC} -m "${MODEL_PATH}" --kvq-key 8 --kvq-val 4 -t 8 -n 32 -p "Hello world" > "${LOG_FILE}" 2>&1 || {
    echo -e "${RED}Test 3 failed with exit code $?${NC}"
    error_exit "$LINENO" "Split precision K8V4 test failed"
}
echo -e "${GREEN}✓ Test 3 completed successfully${NC}"
extract_metal_stats "${LOG_FILE}" "K8V4"
verify_kv_types "${LOG_FILE}" "q8_0" "q4_0"

# Test case 4: Reverse configuration (K4V8 - 4-bit keys, 8-bit values)
echo -e "${BLUE}Test 4: Reverse configuration (K4V8 - 4-bit keys, 8-bit values)${NC}"
LOG_FILE="${RESULTS_DIR}/kvsplit_test_k4v8.log"
echo "Running: ./${MAIN_EXEC} -m ${MODEL_PATH} --kvq-key 4 --kvq-val 8 -t 8 -n 32 -p \"Hello world\""
./${MAIN_EXEC} -m "${MODEL_PATH}" --kvq-key 4 --kvq-val 8 -t 8 -n 32 -p "Hello world" > "${LOG_FILE}" 2>&1 || {
    echo -e "${RED}Test 4 failed with exit code $?${NC}"
    error_exit "$LINENO" "Reverse configuration K4V8 test failed"
}
echo -e "${GREEN}✓ Test 4 completed successfully${NC}"
extract_metal_stats "${LOG_FILE}" "K4V8"
verify_kv_types "${LOG_FILE}" "q4_0" "q8_0"

# Test case 5: Both 4-bit (K4V4 - 4-bit keys, 4-bit values)
echo -e "${BLUE}Test 5: Both 4-bit (K4V4 - 4-bit keys, 4-bit values)${NC}"
LOG_FILE="${RESULTS_DIR}/kvsplit_test_k4v4.log"
echo "Running: ./${MAIN_EXEC} -m ${MODEL_PATH} --kvq 4 -t 8 -n 32 -p \"Hello world\""
./${MAIN_EXEC} -m "${MODEL_PATH}" --kvq 4 -t 8 -n 32 -p "Hello world" > "${LOG_FILE}" 2>&1 || {
    echo -e "${RED}Test 5 failed with exit code $?${NC}"
    error_exit "$LINENO" "K4V4 test failed"
}
echo -e "${GREEN}✓ Test 5 completed successfully${NC}"
extract_metal_stats "${LOG_FILE}" "K4V4"
verify_kv_types "${LOG_FILE}" "q4_0" "q4_0"

# Print summary
print_section "KVSplit Validation Summary"
echo -e "${GREEN}✓ All tests completed successfully!${NC}"
echo "Results saved in: ${RESULTS_DIR}"
echo ""
echo -e "${YELLOW}Memory usage summary:${NC}"
echo "Baseline (FP16): $(grep "KV cache memory:" "${RESULTS_DIR}/kvsplit_test_fp16.log" | awk '{print $4, $5}')"
echo "Q8_0 (--kvq 8): $(grep "KV cache memory:" "${RESULTS_DIR}/kvsplit_test_kvq_q8.log" | awk '{print $4, $5}')"
echo "K8V4 (--kvq-key 8 --kvq-val 4): $(grep "KV cache memory:" "${RESULTS_DIR}/kvsplit_test_k8v4.log" | awk '{print $4, $5}')"
echo "K4V8 (--kvq-key 4 --kvq-val 8): $(grep "KV cache memory:" "${RESULTS_DIR}/kvsplit_test_k4v8.log" | awk '{print $4, $5}')"
echo "K4V4 (--kvq 4): $(grep "KV cache memory:" "${RESULTS_DIR}/kvsplit_test_k4v4.log" | awk '{print $4, $5}')"
echo ""
echo -e "${BLUE}Notes:${NC}"
echo "1. K8V4 (8-bit keys, 4-bit values) typically provides a good balance of memory savings with lower quality loss"
echo "2. K4V8 (4-bit keys, 8-bit values) typically shows more quality degradation as keys are more sensitive to quantization"
echo "3. Results may vary by model size and context length"
echo "4. Memory measurements may show slight differences from theoretical calculations due to 256B page alignment"
echo ""
echo "Run these with longer contexts and different prompts to better evaluate impact on quality"
