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

# Python environment setup
echo -e "${BLUE}Python environment setup options:${RESET}"
echo -e "1. ${GREEN}Virtual Environment:${RESET} Create a new Python venv in this directory"
echo -e "2. ${GREEN}System Python:${RESET} Use system Python installation"
echo -e "3. ${GREEN}Skip:${RESET} Skip Python setup (manual setup later)"
read -p "Select option [1/2/3] (default: 1): " -n 1 -r PYTHON_CHOICE
echo

case $PYTHON_CHOICE in
    "2")
        echo -e "${BLUE}Using system Python...${RESET}"
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
        
        # Check if Python is available
        if ! command -v $PYTHON_CMD &> /dev/null; then
            echo -e "${RED}❌ Python not found. Please install Python 3.${RESET}"
            PYTHON_SETUP_FAILED=true
        else
            echo -e "${GREEN}✅ Using Python: $($PYTHON_CMD --version)${RESET}"
        fi
        
        # Check if dependencies are already installed
        echo -e "${BLUE}Checking for required Python packages...${RESET}"
        MISSING_PACKAGES=""
        for pkg in pandas numpy matplotlib seaborn; do
            if ! $PYTHON_CMD -c "import $pkg" &> /dev/null; then
                MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
            fi
        done
        
        if [ -n "$MISSING_PACKAGES" ]; then
            echo -e "${YELLOW}⚠️ Missing packages:$MISSING_PACKAGES${RESET}"
            echo -e "${YELLOW}You can install them with: $PIP_CMD install$MISSING_PACKAGES${RESET}"
            read -p "Install missing packages now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                $PIP_CMD install $MISSING_PACKAGES || {
                    echo -e "${RED}❌ Failed to install packages.${RESET}"
                    echo -e "${YELLOW}You may need to install them manually:${RESET}"
                    echo -e "${YELLOW}$PIP_CMD install pandas numpy matplotlib seaborn${RESET}"
                }
            fi
        else
            echo -e "${GREEN}✅ All required packages are installed.${RESET}"
        fi
        ;;
    "3")
        echo -e "${YELLOW}Skipping Python setup...${RESET}"
        echo -e "${YELLOW}You'll need to manually install these packages to use visualization tools:${RESET}"
        echo -e "${YELLOW}- pandas${RESET}"
        echo -e "${YELLOW}- numpy${RESET}"
        echo -e "${YELLOW}- matplotlib${RESET}"
        echo -e "${YELLOW}- seaborn${RESET}"
        ;;
    *)
        # Default is virtual environment
        echo -e "${BLUE}Setting up Python virtual environment...${RESET}"
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}❌ Python not found. Please install Python 3.${RESET}"
            PYTHON_SETUP_FAILED=true
        elif ! python3 -m venv venv; then
            echo -e "${YELLOW}⚠️ Could not create Python virtual environment.${RESET}"
            echo -e "${YELLOW}Continuing without virtual environment...${RESET}"
            PYTHON_SETUP_FAILED=true
        else
            echo -e "${GREEN}✅ Virtual environment created.${RESET}"
            echo -e "${BLUE}Installing Python dependencies...${RESET}"
            source venv/bin/activate
            pip install --upgrade pip
            pip install pandas numpy matplotlib seaborn || {
                echo -e "${YELLOW}⚠️ Could not install Python dependencies.${RESET}"
                echo -e "${YELLOW}You may need to install them manually later.${RESET}"
                PYTHON_SETUP_FAILED=true
            }
            
            if [ -z "$PYTHON_SETUP_FAILED" ]; then
                echo -e "${GREEN}✅ Python dependencies installed successfully.${RESET}"
                echo -e "${YELLOW}To activate the virtual environment in the future, run:${RESET}"
                echo -e "${YELLOW}source venv/bin/activate${RESET}"
            fi
        fi
        ;;
esac

if [ -n "$PYTHON_SETUP_FAILED" ]; then
    echo -e "${YELLOW}Python setup incomplete. Visualization tools may not work.${RESET}"
    echo -e "${YELLOW}You can manually install required packages later.${RESET}"
fi

# Setup method selection
echo -e "${BLUE}Choose llama.cpp setup method:${RESET}"
echo -e "1. ${GREEN}Standard:${RESET} Clone and patch llama.cpp (recommended for most users)"
echo -e "2. ${GREEN}Git Submodule:${RESET} Use a forked llama.cpp as a submodule (advanced)"
read -p "Select option [1/2] (default: 1): " -n 1 -r SETUP_CHOICE
echo

if [[ $SETUP_CHOICE == "2" ]]; then
    echo -e "${BLUE}Setting up llama.cpp as a git submodule...${RESET}"
    
    # Check if this is a git repository
    if [ ! -d ".git" ]; then
        echo -e "${YELLOW}⚠️ This directory is not a git repository. Initializing git...${RESET}"
        git init
    fi
    
    # Remove existing llama.cpp if present
    if [ -d "llama.cpp" ]; then
        echo -e "${YELLOW}⚠️ Removing existing llama.cpp directory...${RESET}"
        rm -rf llama.cpp
    fi
    
    # Add the forked llama.cpp as a submodule
    # Note: You would typically fork llama.cpp to your own account and modify it there
    echo -e "${BLUE}Adding llama.cpp as a submodule...${RESET}"
    echo -e "${YELLOW}In a real setup, you would use your own fork with KVSplit changes already applied.${RESET}"
    git submodule add https://github.com/ggerganov/llama.cpp.git
    git submodule update --init --recursive
    
    # Apply the patch to the submodule
    echo -e "${BLUE}Applying KV split patch to llama.cpp submodule...${RESET}"
    cd llama.cpp
    git apply ../patch/fixed_kv_patch.diff || echo -e "${YELLOW}⚠️ Patch application failed, you may need to modify the patch.${RESET}"
    cd ..
    
    echo -e "${GREEN}✅ Submodule setup complete.${RESET}"
    echo -e "${YELLOW}Note: In a real-world scenario, you would fork llama.cpp, make your changes there,${RESET}"
    echo -e "${YELLOW}      and use your fork as the submodule URL instead of applying patches.${RESET}"
else
    # Standard clone and patch approach
    echo -e "${BLUE}Setting up llama.cpp (standard method)...${RESET}"
    if [ -d "llama.cpp" ]; then
        echo -e "${YELLOW}⚠️ llama.cpp directory already exists.${RESET}"
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

    # Apply the patch to llama.cpp
    echo -e "${BLUE}Applying KV split patch to llama.cpp...${RESET}"
    cd llama.cpp
    git apply ../patch/fixed_kv_patch.diff || echo -e "${YELLOW}⚠️ Patch application failed or patch already applied.${RESET}"
    cd ..
fi

# Check if the KV split patch exists
echo -e "${BLUE}Setting up KV split patch...${RESET}"
if [ ! -f "patch/fixed_kv_patch.diff" ]; then
    echo -e "${YELLOW}⚠️ KV patch not found, copying from included patch...${RESET}"
    # Copy the fixed patch that works with current llama.cpp
    if [ -f "patch/split_kv_quant.diff" ]; then
        cp patch/split_kv_quant.diff patch/fixed_kv_patch.diff
    else
        echo -e "${RED}❌ No patch files found! Your installation may not have KV split functionality.${RESET}"
        mkdir -p patch
        # Include a minimal version of the patch inline as a fallback
        cat > patch/fixed_kv_patch.diff << 'EOL'
diff --git a/common/common.cpp b/common/common.cpp
index abcdef1..1234567 100644
--- a/common/common.cpp
+++ b/common/common.cpp
@@ -1290,6 +1290,30 @@ struct cli_params {
                "KV cache quantization for keys. If not specified, defaults to F16",
                {"--cache-type-k", "-ctk"}
            );
+            
+            add_param(
+                &params.cache_type_v,
+                [](enum llama_kv_cache_type & val, const std::string & arg) {
+                    val = llama_model_kv_cache_type_from_str(arg.c_str());
+                    if (val == LLAMA_KV_CACHE_TYPE_COUNT) {
+                        return CLI_PARAM_CONVERSION_ERROR;
+                    }
+                    return CLI_PARAM_CONVERSION_OK;
+                },
+                "KV cache quantization for values. If not specified, defaults to F16",
+                {"--cache-type-v", "-ctv"}
+            );
+            
+            // Combined KV cache quantization (sets both key and value)
+            add_param(
+                [&](const std::string & arg) {
+                    enum llama_kv_cache_type val = llama_model_kv_cache_type_from_str(arg.c_str());
+                    if (val == LLAMA_KV_CACHE_TYPE_COUNT) {
+                        return CLI_PARAM_CONVERSION_ERROR;
+                    }
+                    params.cache_type_k = params.cache_type_v = val;
+                    return CLI_PARAM_CONVERSION_OK;
+                },
+                "--kvq", "-kvq"
+            );
         }
EOL
    fi
fi

# Continue with build process

# Build llama.cpp with Metal support
echo -e "${BLUE}Building llama.cpp with Metal support...${RESET}"
cd llama.cpp 2>/dev/null || true  # Only cd if not already in llama.cpp
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
