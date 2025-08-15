import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

# Directory containing this script
results_dir = Path(__file__).parent
base_name = 'DS_R1_D_Qwen_7B' #DS_R1_D_Qwen_7B
base_dir = results_dir.parent / base_name

# Collect all JSON files, distinguishing between base and adapter runs
all_json_files = []

# Add files from the main results directory (adapter runs)
for f in results_dir.glob('*.json'):
    if not f.parent.name == base_name: # Exclude 'base_vllm' subfolder here
        all_json_files.append((f, 'adapter'))

# Add files from the 'base_vllm' subfolder
for f in base_dir.glob('*.json'):
    all_json_files.append((f, 'base'))

all_records = []

for fpath, run_type in all_json_files:
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            language = data.get('language')
            pass_at_1 = data.get('evaluation_results', {}).get('pass@1')
            # reasoning_trace_count = data.get('reasoning_trace_count')
            compilation_error_count = data.get('compilation_error_count')
            execution_error_count = data.get('execution_error_count')
            
            if language and pass_at_1 is not None:
                record = {
                    'language': language,
                    'pass@1': pass_at_1,
                    # 'reasoning_trace_count': reasoning_trace_count,
                    'compilation_error_count': compilation_error_count,
                    'execution_error_count': execution_error_count,
                    'type': run_type
                }
                if run_type == 'adapter':
                    adapter_path = data.get('adapter_path')
                    if adapter_path:
                        record['iteration'] = adapter_path.strip('/').split('/')[-3]
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
    lang_df = records_df[records_df['language'] == language].copy()
    
    # has_reasoning_trace = 'reasoning_trace_count' in lang_df.columns and not lang_df['reasoning_trace_count'].isnull().all()
    # if has_reasoning_trace:
    #     lang_df['reasoning_trace_count'] = lang_df['reasoning_trace_count'].fillna(0)
    has_reasoning_trace = False

    has_comp_errors = 'compilation_error_count' in lang_df.columns and not lang_df['compilation_error_count'].isnull().all()
    if has_comp_errors:
        lang_df['compilation_error_count'] = lang_df['compilation_error_count'].fillna(0)

    has_exec_errors = 'execution_error_count' in lang_df.columns and not lang_df['execution_error_count'].isnull().all()

    # Remove execution errors for now
    has_exec_errors = False
    
    if has_exec_errors:
        lang_df['execution_error_count'] = lang_df['execution_error_count'].fillna(0)

    # Separate base and adapter dataframes for the current language
    base_lang_df = lang_df[lang_df['type'] == 'base']
    adapter_lang_df = lang_df[lang_df['type'] == 'adapter']

    # Summarize base results
    base_summary = base_lang_df.groupby('iteration')['pass@1'].agg(['min', 'max', 'mean', 'count']).reset_index()
    # if has_reasoning_trace:
    #     base_rtc_summary = base_lang_df.groupby('iteration')['reasoning_trace_count'].agg(mean_rtc='mean').reset_index()
    #     base_summary = pd.merge(base_summary, base_rtc_summary, on='iteration', how='left')
    if has_comp_errors:
        base_cec_summary = base_lang_df.groupby('iteration')['compilation_error_count'].agg(mean_cec='mean').reset_index()
        base_summary = pd.merge(base_summary, base_cec_summary, on='iteration', how='left')
    if has_exec_errors:
        base_eec_summary = base_lang_df.groupby('iteration')['execution_error_count'].agg(mean_eec='mean').reset_index()
        base_summary = pd.merge(base_summary, base_eec_summary, on='iteration', how='left')

    # Summarize adapter results
    adapter_summary = adapter_lang_df.groupby('iteration')['pass@1'].agg(['min', 'max', 'mean', 'count']).reset_index()
    # if has_reasoning_trace:
    #     adapter_rtc_summary = adapter_lang_df.groupby('iteration')['reasoning_trace_count'].agg(mean_rtc='mean').reset_index()
    #     adapter_summary = pd.merge(adapter_summary, adapter_rtc_summary, on='iteration', how='left')
    if has_comp_errors:
        adapter_cec_summary = adapter_lang_df.groupby('iteration')['compilation_error_count'].agg(mean_cec='mean').reset_index()
        adapter_summary = pd.merge(adapter_summary, adapter_cec_summary, on='iteration', how='left')
    if has_exec_errors:
        adapter_eec_summary = adapter_lang_df.groupby('iteration')['execution_error_count'].agg(mean_eec='mean').reset_index()
        adapter_summary = pd.merge(adapter_summary, adapter_eec_summary, on='iteration', how='left')

    # Combine base and adapter summaries for plotting
    combined_summary = pd.concat([base_summary, adapter_summary], ignore_index=True)

    # Sort the combined summary so 'Base' is first, then checkpoints numerically
    def get_sort_key(iteration):
        if iteration == 'Base':
            return -1
        if isinstance(iteration, str) and 'iteration_' in iteration:
            try:
                return int(iteration.split('_')[-1])
            except (ValueError, IndexError):
                return float('inf')
        return float('inf')

    combined_summary['sort_key'] = combined_summary['iteration'].apply(get_sort_key)
    combined_summary = combined_summary.sort_values(by='sort_key').reset_index(drop=True).drop(columns='sort_key')
    
    # Fill NA for count columns for plotting
    count_cols_to_fill = []
    # if has_reasoning_trace:
    #     count_cols_to_fill.append('mean_rtc')
    if has_comp_errors:
        count_cols_to_fill.append('mean_cec')
    if has_exec_errors:
        count_cols_to_fill.append('mean_eec')

    for col in count_cols_to_fill:
        if col not in combined_summary.columns:
            combined_summary[col] = 0
        combined_summary[col] = combined_summary[col].fillna(0)

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

    any_secondary_axis = has_reasoning_trace or has_comp_errors or has_exec_errors
    if any_secondary_axis:
        ax2 = ax.twinx()
        # if has_reasoning_trace:
        #     ax2.plot(x_positions, combined_summary['mean_rtc'], marker='^', color='green', linestyle='--', label='Mean Reasoning Traces')
        #     for i, row in combined_summary.iterrows():
        #         ax2.text(x_positions[i], row['mean_rtc'] + 0.5, f"{row['mean_rtc']:.0f}", ha='center', va='bottom', fontsize=10, color='darkgreen')
        
        if has_comp_errors:
            ax2.plot(x_positions, combined_summary['mean_cec'], marker='x', color='orange', linestyle=':', label='Mean Compilation Errors')
            for i, row in combined_summary.iterrows():
                ax2.text(x_positions[i], row['mean_cec'] + 0.5, f"{row['mean_cec']:.0f}", ha='center', va='bottom', fontsize=10, color='darkorange')
        
        if has_exec_errors:
            ax2.plot(x_positions, combined_summary['mean_eec'], marker='+', color='purple', linestyle='-.', label='Mean Execution Errors')
            for i, row in combined_summary.iterrows():
                ax2.text(x_positions[i], row['mean_eec'] + 0.5, f"{row['mean_eec']:.0f}", ha='center', va='bottom', fontsize=10, color='indigo')
        
        ax2.set_ylabel("Counts")
        
        max_val = 0
        # if has_reasoning_trace: max_val = max(max_val, combined_summary['mean_rtc'].max())
        if has_comp_errors: max_val = max(max_val, combined_summary['mean_cec'].max())
        if has_exec_errors: max_val = max(max_val, combined_summary['mean_eec'].max())
        ax2.set_ylim(0, max_val * 1.5 if max_val > 0 else 10)

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
    # if has_reasoning_trace:
    #     legend_elements.append(
    #         plt.Line2D([0], [0], marker='^', color='w', label='Mean Reasoning Traces', markerfacecolor='green', markersize=8)
    #     )
    if has_comp_errors:
        legend_elements.append(
            plt.Line2D([0], [0], marker='x', color='orange', linestyle='None', label='Mean Compilation Errors', markersize=8)
        )
    if has_exec_errors:
        legend_elements.append(
            plt.Line2D([0], [0], marker='+', color='purple', linestyle='None', label='Mean Execution Errors', markersize=8)
        )
    ax.legend(handles=legend_elements, frameon=True, shadow=True, borderpad=1, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    chart_filename = f"pass_at_1_range_chart_{language}_by_adapter_with_base.pdf"
    chart_filepath = results_dir / chart_filename
    plt.savefig(chart_filepath, dpi=300)
    print(f"Chart for {language} saved to {chart_filepath}")
    plt.close(fig) 