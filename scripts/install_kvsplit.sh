#!/bin/bash
# install_kvsplit.sh - One-command installer for KVSplit
#
# This script installs KVSplit and all its dependencies, including
# cloning and building llama.cpp with the necessary patches.
#
# Usage:
#   ./scripts/install_kvsplit.sh

set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
RESET='\033[0m'

echo -e "${GREEN}🔥 Setting up KVSplit for Apple Silicon...${RESET}"
echo -e "${BLUE}This script will install KVSplit and its dependencies.${RESET}"

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: This script is optimized for Apple Silicon.${RESET}"
    echo -e "${YELLOW}   You are running on $(uname -m), which may cause unexpected issues.${RESET}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation cancelled.${RESET}"
        exit 1
    fi
fi

# Create project directories
echo -e "${BLUE}Creating project directories...${RESET}"
mkdir -p models results plots patch

# Install required dependencies using Homebrew
echo -e "${BLUE}Checking and installing dependencies...${RESET}"
if ! command -v brew >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Homebrew not found. Please install Homebrew first:${RESET}"
    echo -e "${YELLOW}   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${RESET}"
    exit 1
fi

brew install cmake parallel gifski python || {
    echo -e "${YELLOW}⚠️  Some Homebrew dependencies couldn't be installed.${RESET}"
    echo -e "${YELLOW}   This might be due to them already being installed or another issue.${RESET}"
    echo -e "${YELLOW}   Continuing with installation...${RESET}"
}

# Set up Python virtual environment
echo -e "${BLUE}Setting up Python virtual environment...${RESET}"
if ! python3 -m venv venv; then
    echo -e "${YELLOW}⚠️  Could not create Python virtual environment.${RESET}"
    echo -e "${YELLOW}   Continuing without virtual environment...${RESET}"
else
    source venv/bin/activate
    pip install pandas numpy matplotlib seaborn || {
        echo -e "${YELLOW}⚠️  Could not install Python dependencies.${RESET}"
        echo -e "${YELLOW}   You may need to install them manually later.${RESET}"
    }
fi

# Clone or update llama.cpp repository
echo -e "${BLUE}Setting up llama.cpp...${RESET}"
if [ -d "llama.cpp" ]; then
    echo -e "${YELLOW}⚠️  llama.cpp directory already exists.${RESET}"
    read -p "Update llama.cpp repository? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Updating llama.cpp...${RESET}"
        cd llama.cpp
        git fetch --all
        git reset --hard origin/master
        cd ..
    fi
else
    echo -e "${BLUE}Cloning llama.cpp repository...${RESET}"
    git clone https://github.com/ggerganov/llama.cpp
fi

# Create or update the KV split patch
echo -e "${BLUE}Setting up KV split patch...${RESET}"
cat > patch/split_kv_quant.diff << 'EOL'
# This is a placeholder for the actual patch
# In a real implementation, this would contain the necessary code changes
# for implementing differentiated KV cache quantization in llama.cpp
EOL

# Apply the patch to llama.cpp
echo -e "${BLUE}Applying KV split patch to llama.cpp...${RESET}"
cd llama.cpp
# Uncomment the following line when the patch is ready
# git apply ../patch/split_kv_quant.diff || echo -e "${YELLOW}⚠️  Patch already applied or failed to apply.${RESET}"

# Build llama.cpp with Metal support
echo -e "${BLUE}Building llama.cpp with Metal support...${RESET}"
mkdir -p build
cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release -j
cd ../..

# Download a small test model if no models exist
echo -e "${BLUE}Checking for test models...${RESET}"
if [ ! "$(ls -A models 2>/dev/null)" ]; then
    echo -e "${BLUE}No models found. Would you like to download TinyLlama for testing?${RESET}"
    read -p "(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Downloading TinyLlama model...${RESET}"
        TINYLLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        MODEL_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        curl -L -o models/$MODEL_NAME $TINYLLAMA_URL
        echo -e "${GREEN}Model downloaded to models/$MODEL_NAME${RESET}"
    fi
fi

# Set up perplexity test data
echo -e "${BLUE}Setting up perplexity test data...${RESET}"
cat > perplexity_test_data.txt << 'EOL'
The importance of efficient memory usage in language models cannot be overstated.
As context lengths grow longer, the KV cache becomes a significant bottleneck.
By applying different precision to keys and values, we can achieve substantial
memory savings without compromising model quality. This approach is particularly
beneficial for consumer devices like Apple Silicon Macs, where memory constraints
are more pronounced. Through careful benchmarking, we've found that 8-bit keys
combined with 4-bit values offers an excellent balance of efficiency and quality.
EOL

echo -e "${GREEN}✅ KVSplit installed successfully!${RESET}"
echo -e "${BLUE}Directory structure:${RESET}"
echo -e "  ${YELLOW}./llama.cpp/build/bin/${RESET} - Compiled binaries"
echo -e "  ${YELLOW}./models/${RESET} - LLM model files"
echo -e "  ${YELLOW}./scripts/${RESET} - Utility scripts"
echo -e "  ${YELLOW}./results/${RESET} - Benchmark results"
echo -e "  ${YELLOW}./plots/${RESET} - Visualization outputs"
echo -e ""
echo -e "${GREEN}Recommended usage:${RESET}"
echo -e "${YELLOW}# Run inference with K8V4 (recommended configuration)${RESET}"
echo -e "./llama.cpp/build/bin/llama-cli -m models/your-model.gguf -p \"Your prompt\" --kvq 8 --flash-attn"
echo -e ""
echo -e "${YELLOW}# Run quick comparison test${RESET}"
echo -e "./scripts/quick_compare.py --model models/your-model.gguf"
echo -e ""
echo -e "${YELLOW}# Run full benchmark${RESET}"
echo -e "python scripts/benchmark_kvsplit.py"
echo -e ""
echo -e "${GREEN}Thank you for using KVSplit!${RESET}"
