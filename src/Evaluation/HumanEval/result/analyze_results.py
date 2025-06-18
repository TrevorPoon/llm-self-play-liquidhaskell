import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

# Define the path to the results directory relative to the script
results_dir = Path(__file__).parent

# Model and token filter
TARGET_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Note: The user requested 32769, but the example file shows 32768. Using 32768.
TARGET_MAX_NEW_TOKENS = 32768

all_results = []

for fpath in results_dir.glob("*.json"):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON file: {fpath}")

filtered_results = [
    r for r in all_results
    if r.get("model") == TARGET_MODEL
    and r.get("max_new_tokens") == TARGET_MAX_NEW_TOKENS
    and "pass@1" in r.get("evaluation_results", {})
]

if not filtered_results:
    print("No matching results found for the specified model and max_new_tokens.")
else:
    # Prepare data for DataFrame
    df_data = [
        {
            "language": r["language"],
            "pass@1": r["evaluation_results"]["pass@1"]
        }
        for r in filtered_results
    ]
    df = pd.DataFrame(df_data)

    # Calculate min, max, mean, and count of pass@1 for each language
    summary_df = df.groupby("language")["pass@1"].agg(['min', 'max', 'mean', 'count']).reset_index()

    print("Summary Statistics for Pass@1 per Language:")
    print(summary_df)

    # Plotting for a business professional look (min/max/mean line chart)
    plt.style.use('seaborn-v0_8-whitegrid') # A clean, professional grid style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18 # For suptitle
    })
    
    fig, ax = plt.subplots(figsize=(10, 6)) # Reduced figure size for compactness

    # Sort languages for consistent plotting
    summary_df = summary_df.sort_values(by='language')

    # Plot the vertical lines (min to max range)
    # Use a single, subtle color for the lines
    for i, row in summary_df.iterrows():
        ax.vlines(x=i, ymin=row['min'], ymax=row['max'], colors='gray', linestyle='-', linewidth=1.5)

    # Plot min points (black dots)
    ax.scatter(np.arange(len(summary_df['language'])), summary_df['min'], marker='o', color='#1f77b4', s=50, zorder=3, label='Min Pass@1') # Darker blue
    # Plot max points (black dots)
    ax.scatter(np.arange(len(summary_df['language'])), summary_df['max'], marker='o', color='#1f77b4', s=50, zorder=3, label='Max Pass@1') # Darker blue
    # Plot mean points (grey squares)
    ax.scatter(np.arange(len(summary_df['language'])), summary_df['mean'], marker='s', color='#6a9bd8', s=50, zorder=3, label='Mean Pass@1') # Lighter blue

    # Add labels for mean pass@1 rate (percentage, to the right of the square)
    for i, row in summary_df.iterrows():
        mean_percentage = f'{row['mean']:.1%}' # Format as percentage with one decimal place
        ax.text(i + 0.1, row['mean'], mean_percentage, # Add a small offset to the right
                ha='left', va='center', fontsize=12, color='dimgray')

    ax.set_xlabel("Coding Languages")
    ax.set_ylabel("Pass@1 Rate")

    # Determine total runs for the title
    unique_counts = summary_df['count'].unique()
    if len(unique_counts) == 1:
        num_runs_str = f"{int(unique_counts[0])} Trials"
    else:
        # If counts vary, show the range or max count
        num_runs_str = f"Up to {int(summary_df['count'].max())} Trials (Varies per language)"

    ax.set_title(f"Pass@1 Rate for {TARGET_MODEL} (Max New Tokens: {TARGET_MAX_NEW_TOKENS})\n({num_runs_str})")
    
    ax.set_ylim(bottom=0) # Ensure y-axis starts from 0 for better comparison
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability

    # Set x-ticks and labels explicitly after plotting with numerical positions
    ax.set_xticks(np.arange(len(summary_df['language'])))
    ax.set_xticklabels(summary_df['language'], rotation=45, ha='right')

    # Create custom legend handles for min, max, mean dots/squares
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Min Pass@1', markerfacecolor='#1f77b4', markersize=10), # Darker blue
        plt.Line2D([0], [0], marker='o', color='w', label='Max Pass@1', markerfacecolor='#1f77b4', markersize=10), # Darker blue
        plt.Line2D([0], [0], marker='s', color='w', label='Mean Pass@1', markerfacecolor='#6a9bd8', markersize=10), # Lighter blue
    ]
    ax.legend(handles=legend_elements, frameon=True, shadow=True, borderpad=1, loc='upper left') # Professional legend
    
    ax.grid(True, linestyle='--', alpha=0.6) # Subtle grid

    plt.tight_layout()

    # Save the chart as a PNG file
    chart_filename = f"pass_at_1_range_chart_{TARGET_MODEL.replace('/', '_').replace('-', '_')}_max_tokens_{TARGET_MAX_NEW_TOKENS}.png"
    chart_filepath = results_dir / chart_filename
    plt.savefig(chart_filepath, dpi=300) # Save with higher DPI for better quality
    print(f"Chart saved to {chart_filepath}") 