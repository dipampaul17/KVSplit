#!/usr/bin/env python3
"""
quick_compare.py - Runs a quick comparison of different KV quantization settings

This script provides a simple way to compare different KV cache quantization
configurations for llama.cpp models. It shows memory usage, speed, and quality
metrics in an easy-to-understand table format.

Usage:
    python quick_compare.py --model ~/models/your-model.gguf --prompt "Your test prompt"
"""

import argparse
import subprocess
import re
import os
import json
import tempfile
import time
from pathlib import Path
import sys

# ANSI color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
RESET = '\033[0m'

def print_color(color, message):
    """Print colored message to console"""
    print(f"{color}{message}{RESET}")

def create_temp_prompt_file(prompt, min_length=200):
    """Create a temporary file with sufficient prompt text for perplexity testing"""
    # Ensure the prompt is long enough for meaningful perplexity testing
    if len(prompt) < min_length:
        # Extend the prompt with philosophical content
        extension = """
        The concept of memory efficiency in language models is fundamentally important.
        As we process longer contexts, the memory footprint becomes a critical constraint.
        By applying different precision to attention mechanisms, we can achieve significant
        savings without compromising the quality of generated text or understanding.
        This approach is particularly valuable for resource-constrained environments.
        """
        prompt = prompt + extension
    
    # Write to temp file
    fd, temp_path = tempfile.mkstemp(text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(prompt)
    
    return temp_path

def parse_memory_from_output(output):
    """Parse memory usage metrics from llama.cpp output"""
    try:
        # Parse KV cache size
        kv_cache_match = re.search(r'KV cache elements: (\d+).* (\d+(\.\d+)?) MiB', output, re.MULTILINE)
        if kv_cache_match:
            kv_cache_mb = float(kv_cache_match.group(2))
        else:
            kv_cache_mb = None

        # Parse total VRAM usage
        vram_match = re.search(r'VRAM usage: (\d+(\.\d+)?) MiB', output, re.MULTILINE)
        if vram_match:
            vram_mb = float(vram_match.group(1))
        else:
            # Alternative pattern
            vram_match = re.search(r'GPU memory used: (\d+) bytes = (\d+(\.\d+)?) MB', output, re.MULTILINE)
            if vram_match:
                vram_mb = float(vram_match.group(2))
            else:
                vram_mb = None
                
        return kv_cache_mb, vram_mb
    except Exception as e:
        print_color(RED, f"Error parsing memory metrics: {e}")
        return None, None

def parse_speed_from_output(output):
    """Parse speed metrics from llama.cpp output"""
    try:
        # Parse tokens per second
        speed_match = re.search(r'(\d+(\.\d+)?) tokens/sec', output, re.MULTILINE)
        if speed_match:
            tokens_per_sec = float(speed_match.group(1))
            return tokens_per_sec
        
        # Alternative pattern
        speed_match = re.search(r'eval time: (\d+\.\d+) ms \((\d+\.\d+) tokens/sec\)', output, re.MULTILINE)
        if speed_match:
            tokens_per_sec = float(speed_match.group(2))
            return tokens_per_sec
        
        return None
    except Exception as e:
        print_color(RED, f"Error parsing speed metrics: {e}")
        return None

def parse_perplexity(output):
    """Parse perplexity from llama-perplexity output"""
    try:
        perplexity_match = re.search(r'perplexity: (\d+\.\d+)', output, re.MULTILINE | re.IGNORECASE)
        if perplexity_match:
            return float(perplexity_match.group(1))
        
        # Alternative pattern
        perplexity_match = re.search(r'final\s+(?:avg)?\s*perplexity: (\d+\.\d+)', output, re.MULTILINE | re.IGNORECASE)
        if perplexity_match:
            return float(perplexity_match.group(1))
        
        return None
    except Exception as e:
        print_color(RED, f"Error parsing perplexity: {e}")
        return None

def run_comparison(model_path, prompt, seq_len=2048, num_threads=8):
    """Run a comparison of different KV quantization settings"""
    configs = [
        {"name": "FP16", "args": "--ctx-size {seq_len}", "desc": "Baseline (16-bit)"},
        {"name": "K8V8", "args": "--ctx-size {seq_len} --kvq 8", "desc": "8-bit keys & values"},
        {"name": "K8V4", "args": "--ctx-size {seq_len} --kvq-key 8 --kvq-val 4", "desc": "8-bit keys, 4-bit values (RECOMMENDED)"},
        {"name": "K4V8", "args": "--ctx-size {seq_len} --kvq-key 4 --kvq-val 8", "desc": "4-bit keys, 8-bit values"},
        {"name": "K4V4", "args": "--ctx-size {seq_len} --kvq 4", "desc": "4-bit keys & values"},
    ]
    
    # Validate the model path
    model_path = os.path.expanduser(model_path)
    if not os.path.exists(model_path):
        print_color(RED, f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Create a temporary prompt file for perplexity testing
    prompt_file = create_temp_prompt_file(prompt)
    
    # Get the base directory
    base_dir = Path(__file__).parent.parent.absolute()
    llama_cli_path = base_dir / "llama.cpp" / "build" / "bin" / "llama-cli"
    llama_perplexity_path = base_dir / "llama.cpp" / "build" / "bin" / "llama-perplexity"
    
    # Validate the binaries
    if not os.path.exists(llama_cli_path):
        print_color(RED, f"Error: llama-cli binary not found at {llama_cli_path}")
        print_color(YELLOW, "Did you run the install_kvsplit.sh script?")
        sys.exit(1)
    
    if not os.path.exists(llama_perplexity_path):
        print_color(RED, f"Error: llama-perplexity binary not found at {llama_perplexity_path}")
        print_color(YELLOW, "Did you run the install_kvsplit.sh script?")
        sys.exit(1)
    
    results = []
    fp16_perplexity = None  # For calculating relative perplexity
    
    print_color(GREEN, "Running quick comparison of KV cache configurations:")
    print_color(BLUE, f"Model: {model_path}")
    print_color(BLUE, f"Context size: {seq_len} tokens")
    print_color(BLUE, f"Threads: {num_threads}")
    print()
    
    for i, config in enumerate(configs):
        config_name = config["name"]
        print_color(YELLOW, f"[{i+1}/{len(configs)}] Testing {config_name}: {config['desc']}")
        
        try:
            # Format the args string with the sequence length
            args = config["args"].format(seq_len=seq_len)
            
            # Run inference to measure memory and speed
            inference_cmd = f"{llama_cli_path} -m {model_path} {args} -p \"{prompt[:50]}\" -n 50 -t {num_threads} --flash-attn"
            print_color(BLUE, f"Running: {inference_cmd}")
            
            try:
                inference_output = subprocess.check_output(
                    inference_cmd, shell=True, stderr=subprocess.STDOUT
                ).decode('utf-8', errors='ignore')
            except subprocess.CalledProcessError as e:
                inference_output = e.output.decode('utf-8', errors='ignore')
                print_color(RED, f"Command failed with exit code {e.returncode}")
                print_color(RED, inference_output)
                continue
            
            # Run perplexity test
            perplexity_cmd = f"{llama_perplexity_path} -m {model_path} {args} -f {prompt_file} -t {num_threads}"
            print_color(BLUE, f"Running perplexity test: {perplexity_cmd}")
            
            try:
                perplexity_output = subprocess.check_output(
                    perplexity_cmd, shell=True, stderr=subprocess.STDOUT
                ).decode('utf-8', errors='ignore')
            except subprocess.CalledProcessError as e:
                perplexity_output = e.output.decode('utf-8', errors='ignore')
                print_color(RED, f"Perplexity command failed with exit code {e.returncode}")
                print_color(RED, perplexity_output)
                perplexity_output = ""
            
            # Parse metrics
            kv_cache_mb, vram_mb = parse_memory_from_output(inference_output)
            tokens_per_sec = parse_speed_from_output(inference_output)
            perplexity = parse_perplexity(perplexity_output)
            
            # Store FP16 perplexity as baseline
            if config_name == "FP16" and perplexity is not None:
                fp16_perplexity = perplexity
            
            # Calculate perplexity change
            perplexity_change = None
            if fp16_perplexity is not None and perplexity is not None:
                perplexity_change = ((perplexity - fp16_perplexity) / fp16_perplexity) * 100
            
            results.append({
                "Configuration": config_name,
                "Description": config["desc"],
                "KV_Cache_MB": kv_cache_mb,
                "VRAM_MB": vram_mb,
                "Tokens_per_sec": tokens_per_sec,
                "Perplexity": perplexity,
                "Perplexity_Change_Pct": perplexity_change
            })
            
            print_color(GREEN, f"Completed {config_name} test")
            if kv_cache_mb is not None:
                print_color(GREEN, f"  KV Cache: {kv_cache_mb:.2f} MB")
            if vram_mb is not None:
                print_color(GREEN, f"  VRAM: {vram_mb:.2f} MB")
            if tokens_per_sec is not None:
                print_color(GREEN, f"  Speed: {tokens_per_sec:.2f} tokens/sec")
            if perplexity is not None:
                print_color(GREEN, f"  Perplexity: {perplexity:.4f}")
            if perplexity_change is not None:
                change_color = GREEN if perplexity_change < 1.0 else (YELLOW if perplexity_change < 5.0 else RED)
                print_color(change_color, f"  Quality impact: {perplexity_change:+.2f}% vs FP16")
            
            print()
            time.sleep(1)  # Brief pause between tests
            
        except Exception as e:
            print_color(RED, f"Error running {config_name} test: {e}")
    
    # Clean up the temporary file
    try:
        os.unlink(prompt_file)
    except:
        pass
    
    # Calculate savings percentages
    if len(results) > 0 and "FP16" in [r["Configuration"] for r in results]:
        fp16_result = next(r for r in results if r["Configuration"] == "FP16")
        fp16_kv = fp16_result.get("KV_Cache_MB")
        fp16_vram = fp16_result.get("VRAM_MB")
        
        if fp16_kv is not None:
            for result in results:
                kv = result.get("KV_Cache_MB")
                if kv is not None and result["Configuration"] != "FP16":
                    result["KV_Savings_Pct"] = (1 - kv / fp16_kv) * 100
        
        if fp16_vram is not None:
            for result in results:
                vram = result.get("VRAM_MB")
                if vram is not None and result["Configuration"] != "FP16":
                    result["VRAM_Savings_Pct"] = (1 - vram / fp16_vram) * 100
    
    # Display results as a table
    if len(results) > 0:
        print_color(GREEN, "📊 KVSplit Comparison Results:")
        print()
        
        # Header
        print(f"{'Configuration':<12} {'KV Cache':<15} {'VRAM':<15} {'Speed':<15} {'Quality':<15} {'Description':<30}")
        print("-" * 100)
        
        # Rows
        for result in results:
            config = result.get("Configuration", "")
            
            # KV cache column
            kv_cache = f"{result.get('KV_Cache_MB', 'N/A'):.2f} MB" if result.get('KV_Cache_MB') else "N/A"
            if "KV_Savings_Pct" in result:
                kv_cache += f" (-{result['KV_Savings_Pct']:.1f}%)"
            
            # VRAM column
            vram = f"{result.get('VRAM_MB', 'N/A'):.2f} MB" if result.get('VRAM_MB') else "N/A"
            if "VRAM_Savings_Pct" in result:
                vram += f" (-{result['VRAM_Savings_Pct']:.1f}%)"
            
            # Speed column
            speed = f"{result.get('Tokens_per_sec', 'N/A'):.1f} t/s" if result.get('Tokens_per_sec') else "N/A"
            
            # Quality column
            quality = ""
            if result.get('Perplexity') is not None:
                quality = f"{result.get('Perplexity'):.4f}"
                if result.get('Perplexity_Change_Pct') is not None and config != "FP16":
                    quality += f" ({result['Perplexity_Change_Pct']:+.2f}%)"
            else:
                quality = "N/A"
            
            print(f"{config:<12} {kv_cache:<15} {vram:<15} {speed:<15} {quality:<15} {result.get('Description', ''):<30}")
        
        print()
        print_color(GREEN, "Interpretation:")
        print_color(BLUE, "- Lower KV Cache and VRAM values are better (more memory efficient)")
        print_color(BLUE, "- Higher Speed values are better (faster inference)")
        print_color(BLUE, "- Lower Perplexity values and smaller % changes are better (higher quality)")
        print()
        print_color(GREEN, "Recommendation:")
        print_color(YELLOW, "K8V4 (8-bit keys, 4-bit values) typically offers the best balance of memory savings and quality.")
        print()
        
        # Save results as JSON
        try:
            base_dir = Path(__file__).parent.parent.absolute()
            results_dir = base_dir / "results"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            json_path = results_dir / f"quick_compare_{timestamp}.json"
            
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print_color(GREEN, f"Results saved to {json_path}")
        except Exception as e:
            print_color(RED, f"Error saving results: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare KV quantization settings")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--prompt", default="The theory of quantum mechanics explains how particles behave at the atomic and subatomic levels. This counterintuitive framework has revolutionized our understanding of physics.", 
                      help="Test prompt")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length to test")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to use")
    args = parser.parse_args()
    
    run_comparison(args.model, args.prompt, args.seq_len, args.threads)

if __name__ == "__main__":
    main()
