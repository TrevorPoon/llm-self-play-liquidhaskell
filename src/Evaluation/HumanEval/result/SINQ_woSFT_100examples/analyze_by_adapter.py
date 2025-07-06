import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

# Directory containing this script
results_dir = Path(__file__).parent
base_dir = results_dir / 'base'

# Collect all JSON files, distinguishing between base and adapter runs
all_json_files = []

# Add files from the main results directory (adapter runs)
for f in results_dir.glob('*.json'):
    if not f.parent.name == 'base': # Exclude 'base' subfolder here
        all_json_files.append((f, 'adapter'))

# Add files from the 'base' subfolder
for f in base_dir.glob('*.json'):
    all_json_files.append((f, 'base'))

all_records = []

for fpath, run_type in all_json_files:
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            language = data.get('language')
            pass_at_1 = data.get('evaluation_results', {}).get('pass@1')
            
            if language and pass_at_1 is not None:
                record = {
                    'language': language,
                    'pass@1': pass_at_1,
                    'type': run_type
                }
                if run_type == 'adapter':
                    adapter_path = data.get('adapter_path')
                    if adapter_path:
                        record['iteration'] = adapter_path.strip('/').split('/')[-1]
                    else:
                        # If adapter_path is missing for an adapter run, skip or handle as needed
                        print(f"Skipping adapter file {fpath}: missing adapter_path")
                        continue
                else: # 'base' run
                    record['iteration'] = 'Base' # Assign a specific label for base results
                all_records.append(record)
    except Exception as e:
        print(f"Skipping {fpath}: {e}")

if not all_records:
    print("No valid records found.")
    exit(0)

# Convert to DataFrame
records_df = pd.DataFrame(all_records)

for language in records_df['language'].unique():
    lang_df = records_df[records_df['language'] == language]
    
    # Separate base and adapter dataframes for the current language
    base_lang_df = lang_df[lang_df['type'] == 'base']
    adapter_lang_df = lang_df[lang_df['type'] == 'adapter']

    # Summarize base results
    base_summary = base_lang_df.groupby('iteration')['pass@1'].agg(['min', 'max', 'mean', 'count']).reset_index()

    # Summarize adapter results
    adapter_summary = adapter_lang_df.groupby('iteration')['pass@1'].agg(['min', 'max', 'mean', 'count']).reset_index()
    adapter_summary = adapter_summary.sort_values(by='iteration')

    # Combine base and adapter summaries for plotting
    # Ensure 'iteration' is the first column and is consistent
    combined_summary = pd.concat([base_summary, adapter_summary], ignore_index=True)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })
    fig, ax = plt.subplots(figsize=(12, 6)) # Increase figure width to accommodate 'Base'

    # Use numerical positions for x-axis to ensure consistent spacing
    x_positions = np.arange(len(combined_summary['iteration']))

    # Plot min-max vertical lines
    for i, row in combined_summary.iterrows():
        ax.vlines(x=x_positions[i], ymin=row['min'], ymax=row['max'], colors='gray', linestyle='-', linewidth=1.5)
    
    # Plot min points
    ax.scatter(x_positions, combined_summary['min'], marker='o', color='#1f77b4', s=50, zorder=3, label='Min Pass@1')
    # Plot max points
    ax.scatter(x_positions, combined_summary['max'], marker='o', color='#1f77b4', s=50, zorder=3, label='Max Pass@1')
    # Plot mean points
    ax.scatter(x_positions, combined_summary['mean'], marker='s', color='#6a9bd8', s=50, zorder=3, label='Mean Pass@1')

    # Add mean labels
    for i, row in combined_summary.iterrows():
        mean_percentage = f"{row['mean']:.1%}"
        ax.text(x_positions[i], row['mean'] + 0.05, mean_percentage, ha='center', va='bottom', fontsize=12, color='dimgray')

    ax.set_xlabel("Iteration (adapter) / Base")
    ax.set_ylabel("Pass@1 Rate")
    ax.set_title(f"Pass@1 Rate by Adapter Iteration with Base Benchmark for {language}")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(combined_summary['iteration'], rotation=45, ha='right')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Min/Max Pass@1', markerfacecolor='#1f77b4', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Mean Pass@1', markerfacecolor='#6a9bd8', markersize=10),
    ]
    ax.legend(handles=legend_elements, frameon=True, shadow=True, borderpad=1, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    chart_filename = f"pass_at_1_range_chart_{language}_by_adapter_with_base.png"
    chart_filepath = results_dir / chart_filename
    plt.savefig(chart_filepath, dpi=300)
    print(f"Chart for {language} saved to {chart_filepath}")
    plt.close(fig) 