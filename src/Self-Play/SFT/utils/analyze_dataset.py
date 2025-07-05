import argparse
import numpy as np
from datasets import load_from_disk
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import os

def analyze_token_length(dataset_path, model_name_or_path):
    """
    Analyzes the token length of a raw text dataset by tokenizing it on the fly.

    Args:
        dataset_path (str): The path to the processed, untokenized dataset directory.
        model_name_or_path (str): The name or path of the model for loading the tokenizer.
    """
    print(f"Loading tokenizer for '{model_name_or_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and points to the untokenized dataset.")
        return
    
    # Randomly sample 1,000 examples if dataset is larger than that

    if 'train' not in dataset:
        print("Error: 'train' split not found in the dataset.")
        return
        
    train_dataset = dataset['train']

    if len(train_dataset) > 10000:
        print(f"Sampling 10,000 examples from the dataset...")
        train_dataset = train_dataset.shuffle(seed=42).select(range(10000))
    else: 
        print(f"Dataset size is {len(train_dataset)}")

    if 'text' not in train_dataset.column_names:
        print("Error: 'text' column not found. The dataset does not appear to be in the expected format.")
        return

    print("Tokenizing dataset on the fly and calculating token lengths...")
    
    def get_token_length(example):
        return {'length': len(tokenizer(example['text']).input_ids)}

    lengths_dataset = train_dataset.map(
        get_token_length,
        num_proc=os.cpu_count()
    )
    
    lengths = lengths_dataset['length']
    
    if not lengths:
        print("The training dataset is empty or tokenization failed.")
        return

    # --- Calculate Statistics ---
    avg_len = np.mean(lengths)
    min_len = np.min(lengths)
    max_len = np.max(lengths)
    median_len = np.median(lengths)
    p90_len = np.percentile(lengths, 90)
    p95_len = np.percentile(lengths, 95)
    p99_len = np.percentile(lengths, 99)

    print("\n--- Token Length Statistics for 'train' split (from raw text) ---")
    print(f"Number of samples: {len(lengths)}")
    print(f"Average length: {avg_len:.2f}")
    print(f"Median length: {median_len}")
    print(f"Min length: {min_len}")
    print(f"Max length: {max_len}")
    print(f"90th percentile: {p90_len:.2f}")
    print(f"95th percentile: {p95_len:.2f}")
    print(f"99th percentile: {p99_len:.2f}")
    
    print("\nThis analysis suggests a `max_length` around the 90th or 95th percentile to balance coverage and memory usage.")
    print("For example, a max_length of", int(p95_len), "would cover 95% of your training data.")

    # --- Generate and Save Histogram ---
    print("\nGenerating token length histogram...")
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.75)
    plt.title('Distribution of Token Lengths in Training Set (from raw text)')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.axvline(avg_len, color='r', linestyle='dashed', linewidth=2, label=f'Average: {avg_len:.2f}')
    plt.axvline(p95_len, color='g', linestyle='dashed', linewidth=2, label=f'95th Percentile: {p95_len:.2f}')
    plt.legend()
    plt.grid(True)
    
    histogram_path = 'token_length_histogram_from_raw.png'
    plt.savefig(histogram_path)
    print(f"Histogram saved to {histogram_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the token length of a pre-tokenized dataset.")
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        required=True, 
        help="Path to the processed, but untokenized dataset directory."
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help="Model name or path to load the correct tokenizer."
    )
    args = parser.parse_args()
    analyze_token_length(args.dataset_path, args.model_name_or_path) 