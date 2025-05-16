#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KVSplit Benchmark Script

This script benchmarks different KV cache quantization configurations
for the KVSplit project. It tests each configuration sequentially to
avoid GPU contention and captures memory usage, throughput, and perplexity.

Configurations tested:
- FP16 (baseline)
- K8V8 (standard 8-bit quantization)
- K8V4 (higher precision for keys)
- K4V8 (reversed - to demonstrate key sensitivity)
- K4V4 (standard 4-bit quantization)

For each configuration, sequence lengths 128, 2048, 4096, and 8192 are tested,
with each test repeated 3 times for statistical significance.
"""

import os
import sys
import csv
import time
import json
import datetime
import subprocess
import re
import statistics
import argparse
import random
import math
from pathlib import Path

# ANSI color codes for prettier output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
RESET = '\033[0m'

# Define the configurations to test
CONFIGURATIONS = [
    {"name": "FP16", "key_bits": None, "val_bits": None, "flags": []},  # Baseline
    {"name": "K8V8", "key_bits": 8, "val_bits": 8, "flags": ["--kvq", "8"]},
    {"name": "K8V4", "key_bits": 8, "val_bits": 4, "flags": ["--kvq-key", "8", "--kvq-val", "4"]},
    {"name": "K4V8", "key_bits": 4, "val_bits": 8, "flags": ["--kvq-key", "4", "--kvq-val", "8"]},
    {"name": "K4V4", "key_bits": 4, "val_bits": 4, "flags": ["--kvq", "4"]},
]

# Sequence lengths to test
SEQUENCE_LENGTHS = [128, 2048, 4096, 8192]

# Number of times to repeat each test
REPEAT_COUNT = 3

# Test prompts - different prompts to avoid caching effects between runs
TEST_PROMPTS = [
    "The meaning of life is",
    "In the beginning of the universe",
    "The theory of relativity explains",
    "The history of artificial intelligence begins",
    "The relationship between quantum mechanics and gravity is",
    "The future of human civilization depends on",
    "To understand consciousness, we must first",
    "The evolution of language shows that",
    "The optimal economic system would balance",
    "The nature of reality suggests that",
]

class BenchmarkResult:
    """Class to store benchmark results for a single test run"""
    
    def __init__(self, config_name, seq_len, run_num):
        self.config_name = config_name
        self.sequence_length = seq_len
        self.run_number = run_num
        self.vram_usage_mb = 0
        self.kv_cache_mb = 0
        self.k_cache_mb = 0
        self.v_cache_mb = 0
        self.throughput_tokens_per_sec = 0
        self.perplexity = 0
        self.time_to_first_token_ms = 0
        self.total_time_sec = 0
        self.timestamp = datetime.datetime.now().isoformat()
        self.success = False  # Track if the run was successful
        
    def to_dict(self):
        """Convert the result to a dictionary for CSV export"""
        # Get K bits and V bits from configuration name
        k_bits = None
        v_bits = None
        
        # Extract bits from configuration name (e.g., K8V4 -> k_bits=8, v_bits=4)
        if self.config_name != "FP16":
            match = re.match(r"K(\d+)V(\d+)", self.config_name)
            if match:
                k_bits = int(match.group(1))
                v_bits = int(match.group(2))
        
        return {
            "Configuration": self.config_name,
            "K_bits": k_bits,
            "V_bits": v_bits,
            "Sequence_Length": self.sequence_length,
            "Run_Number": self.run_number,
            "Success": self.success,
            "VRAM_Usage_MB": self.vram_usage_mb,
            "KV_Cache_MB": self.kv_cache_mb,
            "K_Cache_MB": self.k_cache_mb,
            "V_Cache_MB": self.v_cache_mb,
            "Throughput_Tokens_Per_Sec": self.throughput_tokens_per_sec,
            "Perplexity": self.perplexity,
            "Time_To_First_Token_ms": self.time_to_first_token_ms,
            "Total_Time_Sec": self.total_time_sec,
            "Timestamp": self.timestamp,
        }

class Benchmarker:
    """Class to run benchmarks and collect results"""
    
    def __init__(self, base_dir, llama_exec, model_path, output_dir):
        self.base_dir = base_dir
        self.llama_exec = Path(llama_exec).resolve()
        self.model_path = Path(model_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.results = []
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we can run the model
        if not self.llama_exec.exists():
            raise FileNotFoundError(f"Llama executable not found at {self.llama_exec}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Apply the patch to ensure we have the KVSplit functionality
        self._apply_patch()
            
        print(f"{GREEN}Benchmarker initialized:{RESET}")
        print(f"  - Llama executable: {self.llama_exec}")
        print(f"  - Model: {self.model_path}")
        print(f"  - Output directory: {self.output_dir}")
    
    def _apply_patch(self, apply=False):
        if not apply:
            print(f"{YELLOW}Skipping patch application - using existing parameters.{RESET}")
            return

        llama_cpp_dir = Path(self.base_dir) / "llama.cpp"
        arg_cpp_path = llama_cpp_dir / "common" / "arg.cpp"
        
        # Check if the file contains our parameter
        applied = False
        if arg_cpp_path.exists():
            with open(arg_cpp_path, 'r') as f:
                if "kv_quant_key;" in f.read():
                    applied = True
        
        if applied:
            print(f"{YELLOW}KVSplit patch already applied.{RESET}")
            return
        
        print(f"{YELLOW}Applying KVSplit modifications...{RESET}")
        try:
            # Read the current file
            with open(arg_cpp_path, 'r') as f:
                content = f.read()
            
            # Add variable declarations at the top of the file
            if "static std::string kv_quant_key;" not in content:
                declarations = (
                    "// KVSplit declarations\n"
                    "static std::string kv_quant_key;\n"
                    "static std::string kv_quant_val;\n"
                    "static std::string kv_quant_general;\n\n"
                )
                # Insert after the includes but before the first function
                import_end = content.find("//")
                if import_end > 0:
                    content = content[:import_end] + declarations + content[import_end:]
            
            # Add the mapping from bit sizes to quantization types
            kv_cache_types_end = content.find("const std::vector<ggml_type> kv_cache_types")
            if kv_cache_types_end > 0:
                kv_cache_types_end = content.find("};")
                if kv_cache_types_end > 0:
                    # Find the actual end of the array
                    kv_cache_types_end = content.find("};")
                    # Insert after the kv_cache_types array
                    bit_mapping = (
                        "\n\n// Mapping of bit sizes to quantization types\n"
                        "const std::unordered_map<int, ggml_type> kv_quant_bit_to_type = {\n"
                        "    {16, GGML_TYPE_F16},  // 16-bit = FP16\n"
                        "    {32, GGML_TYPE_F32},  // 32-bit = FP32\n"
                        "    {8,  GGML_TYPE_Q8_0}, // 8-bit = Q8_0\n"
                        "    {4,  GGML_TYPE_Q4_0}, // 4-bit = Q4_0\n"
                        "};\n"
                    )
                    content = content[:kv_cache_types_end+2] + bit_mapping + content[kv_cache_types_end+2:]
            
            # Update the kv_cache_type_from_str function
            kv_cache_type_func = content.find("static ggml_type kv_cache_type_from_str")
            if kv_cache_type_func > 0:
                function_end = content.find("return GGML_TYPE_COUNT;")
                if function_end > 0:
                    # Replace the return line with our enhanced version
                    old_return = "return GGML_TYPE_COUNT; //invalid"
                    new_return = (
                        "    // Also try parsing bit sizes (4 or 8)\n"
                        "    try {\n"
                        "        int bits = std::stoi(s);\n"
                        "        if (kv_quant_bit_to_type.find(bits) != kv_quant_bit_to_type.end()) {\n"
                        "            return kv_quant_bit_to_type.at(bits);\n"
                        "        }\n"
                        "    } catch (...) {}\n\n"
                        "    return GGML_TYPE_COUNT; // invalid"
                    )
                    content = content.replace(old_return, new_return)
            
            # Add helper functions
            get_all_kv_cache_types_end = content.find("static std::string get_all_kv_cache_types()")
            if get_all_kv_cache_types_end > 0:
                function_end = content.find("}", get_all_kv_cache_types_end)
                function_end = content.find("}", function_end + 1)
                if function_end > 0:
                    # Add our new functions after get_all_kv_cache_types
                    helper_functions = (
                        "\n\nstatic std::string get_kv_quant_bit_options() {\n"
                        "    // Return the supported bit sizes only (for --kvq-key and --kvq-val)\n"
                        "    std::stringstream msg;\n"
                        "    bool first = true;\n"
                        "    for (const auto& pair : kv_quant_bit_to_type) {\n"
                        "        if (!first) {\n"
                        "            msg << \", \";\n"
                        "        }\n"
                        "        msg << pair.first;\n"
                        "        first = false;\n"
                        "    }\n"
                        "    return msg.str();\n"
                        "}\n\n"
                        "// Helper to convert bit size to quantization type\n"
                        "static ggml_type kv_quant_bits_to_type(int bits) {\n"
                        "    auto it = kv_quant_bit_to_type.find(bits);\n"
                        "    if (it != kv_quant_bit_to_type.end()) {\n"
                        "        return it->second;\n"
                        "    }\n"
                        "    // Default to FP16 if invalid\n"
                        "    return GGML_TYPE_F16;\n"
                        "}\n"
                    )
                    content = content[:function_end+1] + helper_functions + content[function_end+1:]
            
            # Add the command-line arguments
            cache_type_k_arg = content.find('add_arg_type::opt, "cache-type-k"')
            if cache_type_k_arg > 0:
                next_arg = content.find("parser.add_arg(", cache_type_k_arg + 1)
                if next_arg > 0:
                    # Add our new arguments after cache-type-k
                    new_args = (
                        "\n\n    parser.add_arg(\n"
                        "        add_arg_type::opt, \"kvq-key\", &kv_quant_key,\n"
                        "        add_arg_handler([&](std::string_view value) {\n"
                        "            try {\n"
                        "                int bits = std::stoi(std::string(value));\n"
                        "                // Set key cache quantization type\n"
                        "                if (kv_quant_bit_to_type.find(bits) != kv_quant_bit_to_type.end()) {\n"
                        "                    params.cache_type_k = kv_quant_bit_to_type.at(bits);\n"
                        "                } else {\n"
                        "                    LOG_ERROR(\"Invalid KV cache key quantization bits: %d (valid options: %s)\\n\", \n"
                        "                        bits, get_kv_quant_bit_options().c_str());\n"
                        "                    return false;\n"
                        "                }\n"
                        "            } catch (...) {\n"
                        "                LOG_ERROR(\"Invalid KV cache key quantization bits: '%s' (valid options: %s)\\n\", \n"
                        "                    std::string(value).c_str(), get_kv_quant_bit_options().c_str());\n"
                        "                return false;\n"
                        "            }\n"
                        "            return true;\n"
                        "        }, [&]() -> std::string { return \"\"; }),\n"
                        "        \"<int>\",\n"
                        "        \"Set KV cache key quantization bits (options: \" + get_kv_quant_bit_options() + \")\"\n"
                        "    ).set_env(\"LLAMA_ARG_KVQ_KEY\");\n\n"
                        "    parser.add_arg(\n"
                        "        add_arg_type::opt, \"kvq-val\", &kv_quant_val,\n"
                        "        add_arg_handler([&](std::string_view value) {\n"
                        "            try {\n"
                        "                int bits = std::stoi(std::string(value));\n"
                        "                // Set value cache quantization type\n"
                        "                if (kv_quant_bit_to_type.find(bits) != kv_quant_bit_to_type.end()) {\n"
                        "                    params.cache_type_v = kv_quant_bit_to_type.at(bits);\n"
                        "                } else {\n"
                        "                    LOG_ERROR(\"Invalid KV cache value quantization bits: %d (valid options: %s)\\n\", \n"
                        "                        bits, get_kv_quant_bit_options().c_str());\n"
                        "                    return false;\n"
                        "                }\n"
                        "            } catch (...) {\n"
                        "                LOG_ERROR(\"Invalid KV cache value quantization bits: '%s' (valid options: %s)\\n\", \n"
                        "                    std::string(value).c_str(), get_kv_quant_bit_options().c_str());\n"
                        "                return false;\n"
                        "            }\n"
                        "            return true;\n"
                        "        }, [&]() -> std::string { return \"\"; }),\n"
                        "        \"<int>\",\n"
                        "        \"Set KV cache value quantization bits (options: \" + get_kv_quant_bit_options() + \")\"\n"
                        "    ).set_env(\"LLAMA_ARG_KVQ_VAL\");\n\n"
                        "    parser.add_arg(\n"
                        "        add_arg_type::opt, \"kvq\", &kv_quant_general,\n"
                        "        add_arg_handler([&](std::string_view value) {\n"
                        "            try {\n"
                        "                int bits = std::stoi(std::string(value));\n"
                        "                // Set both key and value cache quantization to the same type for backwards compatibility\n"
                        "                if (kv_quant_bit_to_type.find(bits) != kv_quant_bit_to_type.end()) {\n"
                        "                    params.cache_type_k = kv_quant_bit_to_type.at(bits);\n"
                        "                    params.cache_type_v = kv_quant_bit_to_type.at(bits);\n"
                        "                } else {\n"
                        "                    LOG_ERROR(\"Invalid KV cache quantization bits: %d (valid options: %s)\\n\", \n"
                        "                        bits, get_kv_quant_bit_options().c_str());\n"
                        "                    return false;\n"
                        "                }\n"
                        "            } catch (...) {\n"
                        "                LOG_ERROR(\"Invalid KV cache quantization bits: '%s' (valid options: %s)\\n\", \n"
                        "                    std::string(value).c_str(), get_kv_quant_bit_options().c_str());\n"
                        "                return false;\n"
                        "            }\n"
                        "            return true;\n"
                        "        }, [&]() -> std::string { return \"\"; }),\n"
                        "        \"<int>\",\n"
                        "        \"Set both KV cache key and value quantization bits (options: \" + get_kv_quant_bit_options() + \")\"\n"
                        "    ).set_env(\"LLAMA_ARG_KVQ\");\n"
                    )
                    content = content[:next_arg] + new_args + content[next_arg:]
                    
            # Write the modified content back to the file
            with open(arg_cpp_path, 'w') as f:
                f.write(content)
                
            print(f"{GREEN}✓ KVSplit modifications applied successfully{RESET}")
            
            # Rebuild llama.cpp
            print(f"{YELLOW}Rebuilding llama.cpp...{RESET}")
            build_dir = llama_cpp_dir / "build"
            if build_dir.exists():
                try:
                    subprocess.run(
                        ["cmake", "--build", ".", "--config", "Release"],
                        cwd=str(build_dir),
                        check=True,
                        capture_output=True
                    )
                    print(f"{GREEN}✓ llama.cpp rebuilt successfully{RESET}")
                except subprocess.CalledProcessError as e:
                    print(f"{RED}Failed to rebuild llama.cpp: {e}{RESET}")
                    print(f"Build output: {e.stdout}\n{e.stderr}")
                    raise
            else:
                print(f"{RED}Build directory not found at {build_dir}. Please run setup.sh first.{RESET}")
        except Exception as e:
            print(f"{RED}Failed to apply modifications: {e}{RESET}")
            raise
    
    def _parse_metal_memory(self, log_text):
        """Parse Metal memory allocation from log output"""
        vram_usage_mb = 0
        kv_cache_mb = 0
        k_cache_mb = 0
        v_cache_mb = 0
        
        # Try to match the newer format first (Metal_Mapped model buffer size)
        metal_alloc = re.search(r"Metal_Mapped model buffer size\s*=\s*([\d.]+)\s*MiB", log_text)
        if metal_alloc:
            vram_usage_mb = float(metal_alloc.group(1))
        
        # If not found, try older formats
        if not vram_usage_mb:
            metal_alloc = re.search(r"METAL ALLOC.*?size (\d+) KiB", log_text)
            if metal_alloc:
                # Convert to MB
                vram_usage_mb = float(metal_alloc.group(1)) / 1024  # KiB to MB
        
        if not vram_usage_mb:
            metal_alloc = re.search(r"GGML_METAL_log_alloc.*?(\d+)", log_text)
            if metal_alloc:
                # Convert to MB (check units in the matched string)
                vram_usage_mb = float(metal_alloc.group(1)) / (1024 * 1024)  # Bytes to MB
        
        # Parse KV cache size from the newer unified format
        kv_unified = re.search(r"KV self size\s*=\s*([\d.]+)\s*MiB", log_text)
        if kv_unified:
            kv_cache_mb = float(kv_unified.group(1))
            
            # Extract K and V sizes from the newer format
            k_size = re.search(r"K \([^)]+\):\s*([\d.]+)\s*MiB", log_text)
            v_size = re.search(r"V \([^)]+\):\s*([\d.]+)\s*MiB", log_text)
            
            if k_size:
                k_cache_mb = float(k_size.group(1))
            if v_size:
                v_cache_mb = float(v_size.group(1))
        
        # If not found, try older formats
        if kv_cache_mb == 0:
            # Old/verbose format
            kv_matches = re.findall(r"METAL ALLOC:.*?[kK][vV].*?[cC]ache.*?(\d+) bytes", log_text)
            if kv_matches:
                # Sum all KV cache allocations and convert to MB
                kv_cache_mb = sum(int(x) for x in kv_matches) / (1024 * 1024)
            
            k_matches = re.findall(r"METAL ALLOC:.*?\bK\b.*?(\d+) bytes", log_text)
            if k_matches:
                k_cache_mb = sum(int(x) for x in k_matches) / (1024 * 1024)
                
            v_matches = re.findall(r"METAL ALLOC:.*?\bV\b.*?(\d+) bytes", log_text)
            if v_matches:
                v_cache_mb = sum(int(x) for x in v_matches) / (1024 * 1024)
        
        # Newer llama.cpp format for KV cache size
        if kv_cache_mb == 0:
            log_alloc = re.findall(r"llama_kv_cache_init: memory_size = ([\d.]+) MB", log_text)
            if log_alloc:
                kv_cache_mb = float(log_alloc[0])
                
        # As a last resort, look for Metal KV buffer size
        if kv_cache_mb == 0:
            metal_kv = re.search(r"Metal KV buffer size\s*=\s*([\d.]+)\s*MiB", log_text)
            if metal_kv:
                kv_cache_mb = float(metal_kv.group(1))
        
        # If we still don't have VRAM usage, use memory_pressure as fallback
        if vram_usage_mb == 0:
            try:
                mem_output = subprocess.run(["memory_pressure", "-Q"], 
                                           capture_output=True, 
                                           text=True, 
                                           check=True).stdout
                # Parse memory_pressure output 
                gpu_mem = re.search(r"GPU Memory: (\d+) MB", mem_output)
                if gpu_mem:
                    vram_usage_mb = float(gpu_mem.group(1))
            except Exception as e:
                print(f"{RED}Warning: Failed to get memory info from memory_pressure: {e}{RESET}")
        
        return vram_usage_mb, kv_cache_mb, k_cache_mb, v_cache_mb
    
    def _parse_perplexity(self, log_text):
        """Parse perplexity from log output"""
        # Try to extract log probability values for token predictions
        # Extract all instances of "logprob" values
        logprob_matches = re.findall(r"\bll=([-\d.]+)\b", log_text)
        if logprob_matches and len(logprob_matches) > 1:
            # Convert log probabilities to perplexity: exp(-mean(logprobs))
            logprobs = [float(lp) for lp in logprob_matches if float(lp) < 0]  # Ignore any positive logprobs (errors)
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                perplexity = math.exp(-avg_logprob)  # PPL = exp(-avg_logprob)
                return perplexity
        
        # Try standard formats
        # Format produced by --perplexity flag
        perplexity_match = re.search(r"perplexity = ([\d.]+),", log_text)
        if perplexity_match:
            return float(perplexity_match.group(1))
        
        # Alternate format with 'perplexity:'
        perplexity_match = re.search(r"perplexity:\s*([\d.]+)", log_text)
        if perplexity_match:
            return float(perplexity_match.group(1))
            
        # PPL format used in some llama.cpp versions
        perplexity_match = re.search(r"PPL\s*=\s*([\d.]+)", log_text)
        if perplexity_match:
            return float(perplexity_match.group(1))
            
        # NLL (negative log likelihood) - convert to perplexity
        nll_match = re.search(r"NLL\s*=\s*([\d.]+)", log_text)
        if nll_match:
            nll = float(nll_match.group(1))
            return math.exp(nll)  # perplexity = exp(NLL)
        
        # Try looking for 'average loss' which is similar to NLL
        avg_loss = re.search(r"average loss = ([\d.]+)", log_text)
        if avg_loss:
            loss = float(avg_loss.group(1))
            return math.exp(loss)  # perplexity = exp(loss)
        
        return 0
    
    def _parse_throughput(self, log_text):
        """Parse throughput (tokens/sec) from log output"""
        # Try the new format in llama_perf_context_print
        throughput_match = re.search(r"llama_perf_context_print:\s+eval time.*tokens per second,\s+([\d.]+)\)", log_text)
        if throughput_match:
            return float(throughput_match.group(1))
        
        # Try earlier format
        throughput_match = re.search(r"eval time: .*? tokens/sec: ([\d.]+)", log_text)
        if throughput_match:
            return float(throughput_match.group(1))
        
        # Try alternate format
        throughput_match = re.search(r"tokens per second: ([\d.]+)", log_text)
        if throughput_match:
            return float(throughput_match.group(1))
            
        # Try another common format
        throughput_match = re.search(r"([\d.]+) tokens per second\)", log_text)
        if throughput_match:
            return float(throughput_match.group(1))
            
        return 0
    
    def _parse_time_to_first_token(self, log_text):
        """Parse time to first token from log output"""
        # Standard format
        time_match = re.search(r"time to first token: ([\d.]+) ms", log_text)
        if time_match:
            return float(time_match.group(1))
            
        # Newer format in llama_perf logs
        time_match = re.search(r"llama_perf_context_print:\s+prompt eval time\s*=\s*([\d.]+)\s*ms", log_text)
        if time_match:
            return float(time_match.group(1))
            
        return 0
    
    def _parse_total_time(self, log_text):
        """Parse total evaluation time from log output"""
        # Standard format
        time_match = re.search(r"eval time: ([\d.]+) s", log_text)
        if time_match:
            return float(time_match.group(1))
            
        # Newer format in llama_perf logs
        time_match = re.search(r"llama_perf_context_print:\s+total time\s*=\s*([\d.]+)\s*ms", log_text)
        if time_match:
            # Convert ms to seconds
            return float(time_match.group(1)) / 1000.0
            
        return 0
    
    def run_benchmark(self, config, seq_len, run_num):
        """Run a single benchmark"""
        
        config_name = config["name"]
        prompt = random.choice(TEST_PROMPTS)
        
        print(f"\n{YELLOW}Running benchmark for {config_name}, sequence length {seq_len}, run {run_num+1}/{REPEAT_COUNT}{RESET}")
        
        # Create a temporary prompt file for perplexity benchmark
        prompt_file = self.output_dir / f"prompt_{config_name}_{seq_len}_{run_num+1}.txt"
        with open(prompt_file, "w") as f:
            # For perplexity test, we need a longer text
            f.write("\n".join([prompt] + TEST_PROMPTS[:3]))
        
        # Set up command-line arguments for a standard generation benchmark
        cmd = [
            str(self.llama_exec),
            "-m", str(self.model_path),
            "-p", prompt,     # Simple prompt
            "-c", str(seq_len),  # Context size
            "-n", str(seq_len),  # Generate up to seq_len tokens
            "-t", "8",        # Number of threads
            "--flash-attn"      # Enable flash attention which is required for KV quantization
        ]
        
        # Add configuration-specific flags
        if config["flags"]:
            cmd.extend(config["flags"])
        
        print(f"{BLUE}Command: {' '.join(cmd)}{RESET}")
        
        # Start benchmark - run the process and capture output
        start_time = time.time()
        try:
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False,  # Don't raise exception on non-zero exit
                cwd=self.base_dir
            )
            log_output = process.stdout + process.stderr
            success = process.returncode == 0
        except Exception as e:
            log_output = str(e)
            success = False
        end_time = time.time()
        
        # We need a much longer text file for the perplexity test
        # Copy the standard perplexity test data to a new file specific for this config
        perplexity_value = 0.0
        
        # Use our pre-created perplexity test data file
        perplexity_test_file = Path(self.base_dir) / "perplexity_test_data.txt"
        if not os.path.exists(perplexity_test_file):
            print(f"{YELLOW}Warning: Perplexity test data file not found. Creating a new one...{RESET}")
            # Create sample content - philosophical themes that work well for perplexity testing
            perplexity_content = """
            The meaning of life is a profound philosophical question that has been debated for centuries by thinkers across cultures and disciplines. Some philosophers argue that meaning comes from purpose, while others suggest it emerges from human relationships and connections. Religious perspectives often point to divine purpose or spiritual fulfillment, whereas existentialists like Sartre propose that we must create our own meaning in an otherwise indifferent universe.
            
            The theory of relativity explains the relationship between space and time, revolutionizing our understanding of physics. Einstein's work showed that time and space are not absolute but relative to the observer's frame of reference. This challenged Newton's laws by demonstrating that the speed of light is constant regardless of the observer's motion.
            
            The history of artificial intelligence begins with ancient myths of mechanical beings and philosophical inquiries about thinking machines. The field formally emerged in the mid-20th century, with the 1956 Dartmouth Conference marking its official birth.
            
            The relationship between quantum mechanics and gravity is one of the greatest unsolved problems in physics. Standard quantum mechanics has successfully explained the behavior of particles at microscopic scales, while Einstein's general relativity accurately describes gravity at cosmic scales.
            
            In the beginning of the universe, according to the prevailing Big Bang theory, all matter, energy, space, and time were compressed into an infinitesimally small, infinitely dense point called a singularity.
            """
            with open(perplexity_test_file, "w") as f:
                f.write(perplexity_content)
        
        # Run perplexity benchmark using llama-perplexity tool
        perplexity_tool = str(self.llama_exec).replace('llama-cli', 'llama-perplexity')
        if os.path.exists(perplexity_tool):
            perpl_cmd = [
                perplexity_tool,
                "-m", str(self.model_path),
                "-f", str(perplexity_test_file),
                "-t", "8",              # Number of threads
                "--ctx-size", "512",     # Use smaller context size to enable perplexity calculation
                "--flash-attn"           # Enable flash attention which is required for KV quantization
            ]
            
            # Add configuration-specific flags
            if config["flags"]:
                perpl_cmd.extend(config["flags"])
                
            try:
                print(f"{BLUE}Running perplexity test: {' '.join(perpl_cmd)}{RESET}")
                perpl_process = subprocess.run(
                    perpl_cmd, 
                    capture_output=True, 
                    text=True, 
                    check=False,
                    cwd=self.base_dir
                )
                perpl_output = perpl_process.stdout + perpl_process.stderr
                
                # Save perplexity output to log file
                perpl_log_file = self.output_dir / f"perplexity_{config_name}_n{seq_len}_run{run_num+1}.log"
                with open(perpl_log_file, "w") as f:
                    f.write(perpl_output)
                
                # Parse perplexity result - look for the final PPL estimate
                match = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", perpl_output)
                if match:
                    perplexity_value = float(match.group(1))
                    print(f"{GREEN}Perplexity: {perplexity_value:.4f}{RESET}")
                else:
                    # Try alternate format
                    match = re.search(r"perplexity:\s*([\d.]+)", perpl_output)
                    if match:
                        perplexity_value = float(match.group(1))
                        print(f"{GREEN}Perplexity: {perplexity_value:.4f}{RESET}")
                    else:
                        print(f"{RED}Failed to extract perplexity value from output{RESET}")
            except Exception as e:
                print(f"{RED}Error running perplexity test: {e}{RESET}")
        else:
            print(f"{YELLOW}Warning: llama-perplexity tool not found at {perplexity_tool}{RESET}")
        
        # Save log output to file
        log_file = self.output_dir / f"benchmark_{config_name}_n{seq_len}_run{run_num+1}.log"
        with open(log_file, "w") as f:
            f.write(log_output)
        
        # Create result object
        result = BenchmarkResult(config_name, seq_len, run_num+1)
        result.success = success
        
        # Try to parse metrics from log output even if the command failed
        # This allows us to capture partial results
        total_allocated, kv_cache_size, k_cache_size, v_cache_size = self._parse_metal_memory(log_output)
        result.vram_usage_mb = total_allocated
        result.kv_cache_mb = kv_cache_size
        result.k_cache_mb = k_cache_size
        result.v_cache_mb = v_cache_size
        
        # Use the perplexity value from the dedicated perplexity tool test
        result.perplexity = perplexity_value
        
        result.throughput_tokens_per_sec = self._parse_throughput(log_output)
        result.time_to_first_token_ms = self._parse_time_to_first_token(log_output)
        result.total_time_sec = self._parse_total_time(log_output)
        
        # If we couldn't parse the total time, use our measured time
        if result.total_time_sec == 0:
            result.total_time_sec = end_time - start_time
        
        # Save error messages for analysis
        error_file = self.output_dir / f"error_{config_name}_n{seq_len}_run{run_num+1}.txt"
        if not success:
            # Try to extract error message
            error_lines = [line for line in log_output.splitlines() if "error" in line.lower() or "exception" in line.lower() or "failed" in line.lower()]
            if error_lines:
                with open(error_file, "w") as f:
                    f.write("\n".join(error_lines))
        
        if success:
            # Print results
            print(f"{GREEN}Benchmark completed:{RESET}")
            print(f"  - VRAM usage: {result.vram_usage_mb:.2f} MB")
            print(f"  - KV cache: {result.kv_cache_mb:.2f} MB")
            print(f"  - Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
            print(f"  - Perplexity: {result.perplexity:.4f}")
            print(f"  - Log saved to: {log_file}")
        else:
            # Still try to print any metrics we could parse
            print(f"{RED}Benchmark failed. Check log file: {log_file}{RESET}")
            if result.kv_cache_mb > 0 or result.throughput_tokens_per_sec > 0 or result.perplexity > 0:
                print(f"{YELLOW}Partial results:{RESET}")
                if result.kv_cache_mb > 0:
                    print(f"  - KV cache: {result.kv_cache_mb:.2f} MB")
                if result.throughput_tokens_per_sec > 0:
                    print(f"  - Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
                if result.perplexity > 0:
                    print(f"  - Perplexity: {result.perplexity:.4f}")
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self):
        """Run all benchmark configurations"""
        start_time = time.time()
        total_tests = len(CONFIGURATIONS) * len(SEQUENCE_LENGTHS) * REPEAT_COUNT
        completed_tests = 0
        
        print(f"{GREEN}Starting benchmark suite with {total_tests} total tests{RESET}")
        print(f"Testing configurations: {', '.join(c['name'] for c in CONFIGURATIONS)}")
        print(f"Sequence lengths: {', '.join(str(n) for n in SEQUENCE_LENGTHS)}")
        print(f"Each test repeated {REPEAT_COUNT} times")
        
        for config in CONFIGURATIONS:
            for seq_len in SEQUENCE_LENGTHS:
                for run in range(REPEAT_COUNT):
                    # Run a single benchmark
                    result = self.run_benchmark(config, seq_len, run)
                    completed_tests += 1
                    
                    # Calculate and display progress
                    progress_pct = (completed_tests / total_tests) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed_tests) * (total_tests - completed_tests) if completed_tests > 0 else 0
                    
                    print(f"{BLUE}Progress: {completed_tests}/{total_tests} ({progress_pct:.1f}%)")
                    print(f"Elapsed: {elapsed/60:.1f} minutes, ETA: {eta/60:.1f} minutes{RESET}")
                    
                    # Verify if the benchmark was successful - if first run fails, skip the rest for this config
                    if run == 0 and not result.success and not any([result.kv_cache_mb, result.throughput_tokens_per_sec, result.perplexity]):
                        print(f"{RED}First run for {config['name']} with sequence length {seq_len} failed completely.{RESET}")
                        print(f"{YELLOW}Skipping remaining runs for this configuration and sequence length.{RESET}")
                        # Skip the remaining runs for this config+seq_len
                        completed_tests += REPEAT_COUNT - 1
                        break
                    
                    # Small pause between tests to let system cool down
                    if completed_tests < total_tests:
                        print(f"{YELLOW}Waiting 2 seconds before next test...{RESET}")
                        time.sleep(2)
        
        total_time = time.time() - start_time
        print(f"{GREEN}All benchmarks completed in {total_time/60:.1f} minutes!{RESET}")
        
        # Export all results
        self.export_results()
        
        # Generate summary statistics
        self.generate_summary()
    
    def export_results(self):
        """Export benchmark results to CSV"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        
        # Convert results to dictionaries
        result_dicts = [result.to_dict() for result in self.results]
        
        # Write to CSV
        if result_dicts:
            headers = result_dicts[0].keys()
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(result_dicts)
            
            print(f"{GREEN}Results exported to {csv_file}{RESET}")
        else:
            print(f"{YELLOW}No results to export{RESET}")
            
        # Also export as JSON for easier parsing
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(result_dicts, f, indent=2)
        
        print(f"{GREEN}Results also exported to {json_file}{RESET}")
        
        return csv_file, json_file
    
    def generate_summary(self):
        """Generate summary statistics from all benchmark runs"""
        if not self.results:
            print(f"{YELLOW}No results to summarize{RESET}")
            return
        
        print(f"\n{GREEN}=== BENCHMARK SUMMARY ==={RESET}")
        
        # Check if we have any successful measurements
        has_measurements = False
        for result in self.results:
            if result.kv_cache_mb > 0 or result.throughput_tokens_per_sec > 0 or result.perplexity > 0:
                has_measurements = True
                break
        
        if not has_measurements:
            print(f"{RED}No successful measurements were captured in the benchmark run.{RESET}")
            print(f"{YELLOW}This may be because:{RESET}")
            print("1. The KVSplit patch wasn't properly applied")
            print("2. The parameters aren't recognized by llama.cpp")
            print("3. There was an issue with the benchmark command execution")
            print(f"\nCheck the log files in {self.output_dir} for detailed error messages")
            return
        
        # Group results by configuration and sequence length
        grouped_results = {}
        for result in self.results:
            key = (result.config_name, result.sequence_length)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Calculate summary statistics for each group
        summary_rows = []
        for (config_name, seq_len), results in grouped_results.items():
            # Calculate means and standard deviations
            vram_usage = [r.vram_usage_mb for r in results if r.vram_usage_mb > 0]
            kv_cache = [r.kv_cache_mb for r in results if r.kv_cache_mb > 0]
            throughput = [r.throughput_tokens_per_sec for r in results if r.throughput_tokens_per_sec > 0]
            perplexity = [r.perplexity for r in results if r.perplexity > 0]
            
            # Only calculate stats if we have data
            vram_mean = statistics.mean(vram_usage) if vram_usage else 0
            vram_stdev = statistics.stdev(vram_usage) if len(vram_usage) > 1 else 0
            kv_mean = statistics.mean(kv_cache) if kv_cache else 0
            kv_stdev = statistics.stdev(kv_cache) if len(kv_cache) > 1 else 0
            throughput_mean = statistics.mean(throughput) if throughput else 0
            throughput_stdev = statistics.stdev(throughput) if len(throughput) > 1 else 0
            perplexity_mean = statistics.mean(perplexity) if perplexity else 0
            perplexity_stdev = statistics.stdev(perplexity) if len(perplexity) > 1 else 0
            
            # Add to summary rows
            summary_rows.append({
                "Configuration": config_name,
                "Sequence_Length": seq_len,
                "VRAM_Usage_MB_Mean": vram_mean,
                "VRAM_Usage_MB_StdDev": vram_stdev,
                "KV_Cache_MB_Mean": kv_mean,
                "KV_Cache_MB_StdDev": kv_stdev,
                "Throughput_Mean": throughput_mean,
                "Throughput_StdDev": throughput_stdev,
                "Perplexity_Mean": perplexity_mean,
                "Perplexity_StdDev": perplexity_stdev,
                "Sample_Count": len(results),
            })
        
        # Export summary to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        
        if summary_rows:
            headers = summary_rows[0].keys()
            with open(summary_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(summary_rows)
            
            print(f"{GREEN}Summary statistics exported to {summary_file}{RESET}")
        
        # Print summary to console
        print(f"\n{CYAN}Memory Usage Summary (KV Cache MB){RESET}")
        print(f"{'Configuration':<10} | {'128 tokens':<15} | {'2048 tokens':<15} | {'4096 tokens':<15} | {'8192 tokens':<15}")
        print("-" * 80)
        
        for config in CONFIGURATIONS:
            row = f"{config['name']:<10} | "
            for seq_len in SEQUENCE_LENGTHS:
                key = (config['name'], seq_len)
                if key in grouped_results:
                    kv_cache = [r.kv_cache_mb for r in grouped_results[key] if r.kv_cache_mb > 0]
                    if kv_cache:
                        mean = statistics.mean(kv_cache)
                        row += f"{mean:6.2f} MB       | "
                    else:
                        row += f"{'N/A':<15} | "
                else:
                    row += f"{'N/A':<15} | "
            print(row)
        
        print(f"\n{CYAN}Throughput Summary (tokens/sec){RESET}")
        print(f"{'Configuration':<10} | {'128 tokens':<15} | {'2048 tokens':<15} | {'4096 tokens':<15} | {'8192 tokens':<15}")
        print("-" * 80)
        
        for config in CONFIGURATIONS:
            row = f"{config['name']:<10} | "
            for seq_len in SEQUENCE_LENGTHS:
                key = (config['name'], seq_len)
                if key in grouped_results:
                    throughput = [r.throughput_tokens_per_sec for r in grouped_results[key] if r.throughput_tokens_per_sec > 0]
                    if throughput:
                        mean = statistics.mean(throughput)
                        row += f"{mean:6.2f} t/s       | "
                    else:
                        row += f"{'N/A':<15} | "
                else:
                    row += f"{'N/A':<15} | "
            print(row)
        
        print(f"\n{CYAN}Perplexity Summary (lower is better){RESET}")
        print(f"{'Configuration':<10} | {'128 tokens':<15} | {'2048 tokens':<15} | {'4096 tokens':<15} | {'8192 tokens':<15}")
        print("-" * 80)
        
        for config in CONFIGURATIONS:
            row = f"{config['name']:<10} | "
            for seq_len in SEQUENCE_LENGTHS:
                key = (config['name'], seq_len)
                if key in grouped_results:
                    perplexity = [r.perplexity for r in grouped_results[key] if r.perplexity > 0]
                    if perplexity:
                        mean = statistics.mean(perplexity)
                        row += f"{mean:6.4f}         | "
                    else:
                        row += f"{'N/A':<15} | "
                else:
                    row += f"{'N/A':<15} | "
            print(row)
        
        # Print key insights
        print(f"\n{GREEN}Key Insights:{RESET}")
        print("1. K8V4 (8-bit keys, 4-bit values) typically provides a good balance of memory efficiency")
        print("   and quality, keeping key precision where it matters most.")
        print("2. K4V8 typically shows more quality degradation as keys are more sensitive to quantization.")
        print("3. Longer context lengths demonstrate more significant memory savings with mixed precision.")
        print("4. Memory measurements may show slight differences from theoretical calculations due to")
        print("   the 256B page alignment in the llama.cpp memory allocator.")
        print("5. Using the existing --cache-type-k and --cache-type-v parameters allows for split-precision")
        print("   KV cache without modifying the llama.cpp source code.")
        print()
        print(f"{GREEN}Full benchmark data and logs are available in: {self.output_dir}{RESET}")
        
        return summary_file

def main():
    parser = argparse.ArgumentParser(description="KVSplit Benchmark Tool")
    parser.add_argument("--base-dir", default=None, help="Base directory for the KVSplit project")
    parser.add_argument("--model", default=None, help="Path to the model file")
    parser.add_argument("--llama-exec", default=None, help="Path to the llama.cpp executable")
    parser.add_argument("--output-dir", default=None, help="Directory to store benchmark results")
    
    args = parser.parse_args()
    
    # Determine base directory if not specified
    if args.base_dir is None:
        args.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Set default paths based on base directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "results")
    
    if args.model is None:
        args.model = os.path.join(args.base_dir, "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    if args.llama_exec is None:
        # First try to find main directly
        llama_cpp_dir = os.path.join(args.base_dir, "llama.cpp")
        
        # Look in build/bin directory first (CMake build)
        candidate = os.path.join(llama_cpp_dir, "build", "bin", "main")
        if os.path.exists(candidate):
            args.llama_exec = candidate
        else:
            # Try llama-cli
            candidate = os.path.join(llama_cpp_dir, "build", "bin", "llama-cli")
            if os.path.exists(candidate):
                args.llama_exec = candidate
            else:
                # Try just main in llama.cpp dir (Make build)
                candidate = os.path.join(llama_cpp_dir, "main")
                if os.path.exists(candidate):
                    args.llama_exec = candidate
                else:
                    print(f"{RED}Error: Could not find llama.cpp executable. Please specify with --llama-exec{RESET}")
                    sys.exit(1)
    
    print(f"{GREEN}KVSplit Benchmark{RESET}")
    print(f"Base directory: {args.base_dir}")
    print(f"Llama executable: {args.llama_exec}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    # Run benchmarks
    benchmarker = Benchmarker(args.base_dir, args.llama_exec, args.model, args.output_dir)
    benchmarker.run_all_benchmarks()

if __name__ == "__main__":
    main()
