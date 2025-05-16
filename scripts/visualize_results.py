#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Script for KVSplit Benchmarks

This script generates publication-quality plots from the benchmark data,
showing memory usage, performance impact, quality impact, and key-value sensitivity.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use colorblind-friendly style
plt.style.use("tableau-colorblind10")

# Constants
OUTPUT_DIR = Path("../plots")
RESULTS_DIR = Path("../results")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_latest_results():
    """Load the most recent benchmark results CSV file"""
    # Find the most recent benchmark results file
    result_files = glob.glob(str(RESULTS_DIR / "benchmark_results_*.csv"))
    if not result_files:
        raise FileNotFoundError("No benchmark result files found")
    
    # Sort by modification time, newest first
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"Using benchmark data from: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file)
    return df

def prepare_data(df):
    """Process the benchmark data for visualization"""
    # Group by configuration and sequence length, averaging across runs
    grouped = df.groupby(['Configuration', 'Sequence_Length']).agg({
        'VRAM_Usage_MB': 'mean',
        'KV_Cache_MB': 'mean',
        'Throughput_Tokens_Per_Sec': 'mean',
        'Perplexity': 'mean',
        'Success': 'mean'  # How many runs succeeded
    }).reset_index()
    
    # Calculate the baseline FP16 values for each sequence length
    fp16_baseline = grouped[grouped['Configuration'] == 'FP16'].copy()
    
    # Create lookup dictionaries for baseline values
    vram_baseline = dict(zip(fp16_baseline['Sequence_Length'], fp16_baseline['VRAM_Usage_MB']))
    kv_baseline = dict(zip(fp16_baseline['Sequence_Length'], fp16_baseline['KV_Cache_MB']))
    perplexity_baseline = fp16_baseline['Perplexity'].mean()
    
    # Add percent savings and perplexity change columns
    grouped['VRAM_Savings_Pct'] = grouped.apply(
        lambda row: 0 if row['Configuration'] == 'FP16' else 
        (1 - row['VRAM_Usage_MB'] / vram_baseline[row['Sequence_Length']]) * 100, 
        axis=1
    )
    
    grouped['KV_Savings_Pct'] = grouped.apply(
        lambda row: 0 if row['Configuration'] == 'FP16' else 
        (1 - row['KV_Cache_MB'] / kv_baseline[row['Sequence_Length']]) * 100, 
        axis=1
    )
    
    grouped['Perplexity_Change_Pct'] = grouped.apply(
        lambda row: ((row['Perplexity'] - perplexity_baseline) / perplexity_baseline) * 100,
        axis=1
    )
    
    return grouped

def plot_memory_usage(df):
    """Create a bar chart showing memory usage by configuration and sequence length"""
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart for KV cache size
    ax = sns.barplot(
        data=df, 
        x='Configuration', 
        y='KV_Cache_MB', 
        hue='Sequence_Length',
        palette='viridis'
    )
    
    # Add title and labels
    plt.title('KV Cache Memory Usage by Configuration', fontsize=16)
    plt.ylabel('KV Cache Size (MB)', fontsize=14)
    plt.xlabel('Configuration', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust legend
    plt.legend(title='Sequence Length', fontsize=12, title_fontsize=13)
    
    # Add annotations showing % savings vs FP16 for a specific sequence length
    # Choose the longest sequence length for annotations
    longest_seq = df['Sequence_Length'].max()
    long_seq_data = df[df['Sequence_Length'] == longest_seq]
    
    # Get baseline VRAM for FP16
    fp16_kv = long_seq_data[long_seq_data['Configuration'] == 'FP16']['KV_Cache_MB'].values[0]
    
    # Annotate each non-FP16 bar with savings percentage
    bars = [patch for i, patch in enumerate(ax.patches) 
            if i % len(df['Sequence_Length'].unique()) == len(df['Sequence_Length'].unique()) - 1]
    
    configs = long_seq_data['Configuration'].unique()
    for i, bar in enumerate(bars):
        if i < len(configs) and configs[i] != 'FP16':
            config_kv = long_seq_data[long_seq_data['Configuration'] == configs[i]]['KV_Cache_MB'].values[0]
            savings = (1 - config_kv / fp16_kv) * 100
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 5, 
                f'{savings:.1f}%', 
                ha='center', 
                fontsize=11, 
                fontweight='bold',
                color='green'
            )
    
    # Add a note about the annotations
    plt.annotate(
        f'Percentages show memory savings\ncompared to FP16 for {longest_seq} tokens',
        xy=(0.5, 0.97),
        xycoords='figure fraction',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kv_cache_memory_usage.png', dpi=200, bbox_inches="tight")
    plt.close()

def plot_performance(df):
    """Create a bar chart showing throughput by configuration and sequence length"""
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart for throughput
    ax = sns.barplot(
        data=df, 
        x='Configuration', 
        y='Throughput_Tokens_Per_Sec', 
        hue='Sequence_Length',
        palette='viridis'
    )
    
    # Add title and labels
    plt.title('Inference Speed by Configuration', fontsize=16)
    plt.ylabel('Throughput (Tokens per second)', fontsize=14)
    plt.xlabel('Configuration', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Adjust legend
    plt.legend(title='Sequence Length', fontsize=12, title_fontsize=13)
    
    # Calculate average change vs FP16 for configurations
    configs = df['Configuration'].unique()
    seq_lengths = df['Sequence_Length'].unique()
    
    # Average improvement in throughput compared to FP16
    improvements = {}
    for config in configs:
        if config == 'FP16':
            continue
        improvement_sum = 0
        for seq_len in seq_lengths:
            fp16_throughput = df[(df['Configuration'] == 'FP16') & 
                                (df['Sequence_Length'] == seq_len)]['Throughput_Tokens_Per_Sec'].values[0]
            config_throughput = df[(df['Configuration'] == config) & 
                                  (df['Sequence_Length'] == seq_len)]['Throughput_Tokens_Per_Sec'].values[0]
            improvement_pct = ((config_throughput / fp16_throughput) - 1) * 100
            improvement_sum += improvement_pct
        improvements[config] = improvement_sum / len(seq_lengths)
    
    # Annotate with the average improvement
    y_max = df['Throughput_Tokens_Per_Sec'].max() * 1.05
    for i, config in enumerate(configs[1:], 1):  # Skip FP16
        plt.annotate(
            f"{improvements[config]:.1f}% vs FP16",
            xy=(i, y_max * 0.95),
            ha='center',
            fontsize=11,
            fontweight='bold',
            color='green' if improvements[config] > 0 else 'red'
        )
    
    # Add a note about the annotations
    plt.annotate(
        'Percentages show average throughput\nimprovement compared to FP16',
        xy=(0.5, 0.97),
        xycoords='figure fraction',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'inference_speed.png', dpi=200, bbox_inches="tight")
    plt.close()

def plot_quality_impact(df):
    """Create a bar chart showing perplexity change vs FP16"""
    # Get average perplexity per configuration (averaging across sequence lengths)
    perplexity_by_config = df.groupby('Configuration')['Perplexity'].mean().reset_index()
    baseline = perplexity_by_config[perplexity_by_config['Configuration'] == 'FP16']['Perplexity'].values[0]
    
    # Calculate perplexity change vs baseline
    perplexity_by_config['Perplexity_Change'] = ((perplexity_by_config['Perplexity'] - baseline) / baseline) * 100
    
    plt.figure(figsize=(10, 6))
    
    # Plot the perplexity change (excluding FP16)
    non_fp16 = perplexity_by_config[perplexity_by_config['Configuration'] != 'FP16']
    bars = plt.bar(
        non_fp16['Configuration'], 
        non_fp16['Perplexity_Change'],
        color=sns.color_palette("Reds_r", len(non_fp16))
    )
    
    # Add title and labels
    plt.title('Quality Impact: Perplexity Change vs FP16', fontsize=16)
    plt.ylabel('Perplexity Change (%)', fontsize=14)
    plt.xlabel('Configuration', fontsize=14)
    plt.axhline(y=0, color='blue', linestyle='-', alpha=0.3, label='FP16 Baseline')
    
    # Add a red line for 5% degradation threshold
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Degradation Threshold')
    
    # Add annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.1 if height > 0 else height - 0.3,
            f'{height:.2f}%',
            ha='center',
            fontsize=12,
            fontweight='bold',
            color='black'
        )
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add annotation explaining the implications
    plt.annotate(
        'Lower is better. Values below 5% generally\nindicate minimal quality degradation.',
        xy=(0.5, 0.97),
        xycoords='figure fraction',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'perplexity_change.png', dpi=200, bbox_inches="tight")
    plt.close()

def plot_key_vs_value_sensitivity(df):
    """Create a heatmap showing sensitivity to key vs value precision"""
    # Create a dataframe for the heatmap
    configs = ['FP16', 'K8V8', 'K8V4', 'K4V8', 'K4V4']
    k_bits = [16, 8, 8, 4, 4]
    v_bits = [16, 8, 4, 8, 4]
    perplexity_avg = df.groupby('Configuration')['Perplexity'].mean().reset_index()
    
    # Create a lookup for perplexity values
    perplexity_lookup = dict(zip(perplexity_avg['Configuration'], perplexity_avg['Perplexity']))
    perplexity_values = [perplexity_lookup.get(config, np.nan) for config in configs]
    
    # Calculate perplexity change
    fp16_perplexity = perplexity_lookup['FP16']
    perplexity_change = [((p - fp16_perplexity) / fp16_perplexity) * 100 for p in perplexity_values]
    
    # Create dataframe for heatmap
    sensitivity_data = pd.DataFrame({
        'Config': configs,
        'K_bits': k_bits,
        'V_bits': v_bits,
        'Perplexity': perplexity_values,
        'Perplexity_Change_Pct': perplexity_change
    })
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Heatmap of perplexity values by K/V bit precision
    pivot1 = sensitivity_data.pivot(index='K_bits', columns='V_bits', values='Perplexity')
    sns.heatmap(
        pivot1, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis_r',  # Lower is better for perplexity
        ax=ax1,
        cbar_kws={'label': 'Perplexity'}
    )
    ax1.set_title('Perplexity by Key/Value Precision', fontsize=14)
    ax1.set_xlabel('Value Bits', fontsize=12)
    ax1.set_ylabel('Key Bits', fontsize=12)
    
    # Plot 2: Heatmap of perplexity change percentage
    pivot2 = sensitivity_data.pivot(index='K_bits', columns='V_bits', values='Perplexity_Change_Pct')
    sns.heatmap(
        pivot2, 
        annot=True, 
        fmt='.2f', 
        cmap='RdYlGn_r',  # Red for worse, green for better
        ax=ax2,
        cbar_kws={'label': 'Perplexity Change (%)'}
    )
    ax2.set_title('Perplexity Change vs FP16 (%)', fontsize=14)
    ax2.set_xlabel('Value Bits', fontsize=12)
    ax2.set_ylabel('Key Bits', fontsize=12)
    
    # Add overall title
    plt.suptitle('Key vs Value Precision Sensitivity', fontsize=16)
    
    # Add annotation explaining the key findings
    fig.text(
        0.5, 0.02,
        'Key precision (rows) has a larger impact on quality than value precision (columns).\nK8V4 offers an excellent balance of quality and memory efficiency.',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / 'key_value_sensitivity.png', dpi=200, bbox_inches="tight")
    plt.close()

def plot_memory_vs_quality(df):
    """Create a scatter plot showing the tradeoff between memory usage and quality"""
    # Average across sequence lengths
    avg_by_config = df.groupby('Configuration').agg({
        'KV_Cache_MB': 'mean',
        'Perplexity': 'mean',
        'KV_Savings_Pct': 'mean',
        'Perplexity_Change_Pct': 'mean'
    }).reset_index()
    
    # Create custom color map for the configurations
    colors = {
        'FP16': 'blue',
        'K8V8': 'green',
        'K8V4': 'purple',
        'K4V8': 'orange',
        'K4V4': 'red'
    }
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    for config in avg_by_config['Configuration']:
        data = avg_by_config[avg_by_config['Configuration'] == config]
        plt.scatter(
            data['KV_Savings_Pct'], 
            data['Perplexity_Change_Pct'],
            s=200,  # Size
            c=colors[config],  # Color
            label=config,
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add config label to each point
        plt.annotate(
            config,
            xy=(data['KV_Savings_Pct'].values[0], data['Perplexity_Change_Pct'].values[0]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )
    
    # Add quadrant labels
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Quality Threshold')
    plt.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='50% Memory Savings Threshold')
    
    # Add shaded quadrants with labels
    plt.fill_between(
        [-10, 50], 5, 15, color='red', alpha=0.1
    )
    plt.fill_between(
        [50, 100], 5, 15, color='orange', alpha=0.1
    )
    plt.fill_between(
        [-10, 50], -10, 5, color='yellow', alpha=0.1
    )
    plt.fill_between(
        [50, 100], -10, 5, color='green', alpha=0.1
    )
    
    # Add quadrant labels
    plt.text(25, 10, "Low Savings, Worse Quality", ha='center', fontsize=10)
    plt.text(75, 10, "High Savings, Worse Quality", ha='center', fontsize=10)
    plt.text(25, 2, "Low Savings, Better Quality", ha='center', fontsize=10)
    plt.text(75, 2, "High Savings, Better Quality", ha='center', fontsize=10)
    
    # Add title and labels
    plt.title('Memory Savings vs Quality Tradeoff', fontsize=16)
    plt.xlabel('KV Cache Memory Savings (%)', fontsize=14)
    plt.ylabel('Perplexity Change vs FP16 (%)', fontsize=14)
    plt.xlim(-10, 100)
    plt.ylim(-2, 15)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add annotation explaining the optimal region
    plt.annotate(
        'Optimal configurations maximize memory savings\nwhile minimizing perplexity increase.',
        xy=(0.5, 0.97),
        xycoords='figure fraction',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'memory_vs_quality.png', dpi=200, bbox_inches="tight")
    plt.close()

def create_summary_table(df):
    """Create a summary table visual showing the key metrics for each configuration"""
    # Average across sequence lengths
    avg_by_config = df.groupby('Configuration').agg({
        'KV_Cache_MB': 'mean',
        'Throughput_Tokens_Per_Sec': 'mean',
        'Perplexity': 'mean',
        'KV_Savings_Pct': 'mean',
        'Perplexity_Change_Pct': 'mean'
    }).reset_index()
    
    # Add a column for throughput change
    fp16_throughput = avg_by_config[avg_by_config['Configuration'] == 'FP16']['Throughput_Tokens_Per_Sec'].values[0]
    avg_by_config['Throughput_Change_Pct'] = ((avg_by_config['Throughput_Tokens_Per_Sec'] / fp16_throughput) - 1) * 100
    
    # Sort configurations in a specific order
    order = ['FP16', 'K8V8', 'K8V4', 'K4V8', 'K4V4']
    avg_by_config['Order'] = avg_by_config['Configuration'].map({k: i for i, k in enumerate(order)})
    avg_by_config = avg_by_config.sort_values('Order').drop('Order', axis=1)
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Define table data
    table_data = [
        avg_by_config['Configuration'].tolist(),
        [f"{x:.2f} MB" for x in avg_by_config['KV_Cache_MB']],
        [f"{x:.1f}%" for x in avg_by_config['KV_Savings_Pct']],
        [f"{x:.0f}" for x in avg_by_config['Throughput_Tokens_Per_Sec']],
        [f"{x:+.1f}%" for x in avg_by_config['Throughput_Change_Pct']],
        [f"{x:.2f}" for x in avg_by_config['Perplexity']],
        [f"{x:+.2f}%" for x in avg_by_config['Perplexity_Change_Pct']]
    ]
    
    # Define row labels
    row_labels = [
        'Configuration',
        'Avg KV Cache (MB)',
        'Memory Savings',
        'Throughput (t/s)',
        'Throughput vs FP16',
        'Perplexity',
        'Quality Impact'
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.15] * len(avg_by_config)
    )
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color cells based on values
    for i in range(len(row_labels)):
        for j in range(len(avg_by_config)):
            cell = table[(i, j)]
            
            # Memory savings - green for higher savings
            if i == 2 and j > 0:  # KV_Savings_Pct row, excluding FP16
                savings = avg_by_config.iloc[j]['KV_Savings_Pct']
                intensity = min(savings / 80, 1)  # Normalize to [0, 1]
                cell.set_facecolor((1 - intensity, 1, 1 - intensity))
            
            # Throughput - green for better, red for worse
            elif i == 4 and j > 0:  # Throughput_Change_Pct row, excluding FP16
                change = avg_by_config.iloc[j]['Throughput_Change_Pct']
                if change > 0:
                    intensity = min(change / 20, 1)  # Normalize to [0, 1]
                    cell.set_facecolor((1 - intensity, 1, 1 - intensity))
                else:
                    intensity = min(abs(change) / 20, 1)  # Normalize to [0, 1]
                    cell.set_facecolor((1, 1 - intensity, 1 - intensity))
            
            # Perplexity - red for worse (higher values)
            elif i == 6 and j > 0:  # Perplexity_Change_Pct row, excluding FP16
                change = avg_by_config.iloc[j]['Perplexity_Change_Pct']
                if change > 0:
                    intensity = min(change / 10, 1)  # Normalize to [0, 1]
                    cell.set_facecolor((1, 1 - intensity, 1 - intensity))
                else:
                    intensity = min(abs(change) / 10, 1)  # Normalize to [0, 1]
                    cell.set_facecolor((1 - intensity, 1, 1 - intensity))
    
    # Add title
    plt.suptitle('KVSplit Configuration Summary', fontsize=16)
    
    # Add annotation
    plt.figtext(
        0.5, 0.01,
        'Values are averaged across all sequence lengths. Green indicates better performance, red indicates worse.',
        ha='center',
        fontsize=12
    )
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / 'configuration_summary.png', dpi=200, bbox_inches="tight")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    # Load data
    try:
        df = load_latest_results()
        df_processed = prepare_data(df)
        
        # Generate plots
        plot_memory_usage(df_processed)
        plot_performance(df_processed)
        plot_quality_impact(df_processed)
        plot_key_vs_value_sensitivity(df_processed)
        plot_memory_vs_quality(df_processed)
        create_summary_table(df_processed)
        
        print(f"Visualizations saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
