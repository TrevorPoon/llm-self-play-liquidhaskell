# Script to process, format, and tokenize the synthetically generated reasoning dataset.
import os
import argparse
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer

def format_prompt(example: dict) -> dict:
    """
    Formats the instruction, reasoning, and code from a dataset example into a structured
    text block suitable for supervised fine-tuning (SFT) of a reasoning model.

    This format includes a system prompt, the user's instruction, and a response
    that contains both the reasoning (<think> block) and the final Haskell code.
    """
    system_prompt = "You are a helpful and expert Haskell programmer. Please think step-by-step and provide the Haskell code to solve the user's request."
    
    instruction = example.get("instruction", "").strip()
    reasoning = example.get("reasoning", "").strip()
    code = example.get("code", "").strip()

    # Chat-like format for SFT
    return {
        "text": f"""{system_prompt}

### Instruction:
{instruction}

### Response:
<think>
{reasoning}
</think>
```haskell
{code}
```"""
    }

def main():
    parser = argparse.ArgumentParser(description="Process and tokenize the synthetic reasoning dataset for SFT.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the raw synthetic reasoning dataset directory.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model for tokenization.")
    parser.add_argument('--processed_output_dir', type=str, default='../data/sft_processed_reasoning_dataset', help="Directory to save the processed text-only dataset.")
    parser.add_argument('--tokenized_output_dir', type=str, default='../data/sft_tokenized_reasoning_dataset', help="Directory to save the final tokenized dataset.")
    parser.add_argument('--max_length', type=int, default=32768, help="Max sequence length for tokenization.")
    
    # Interleaved splitting arguments
    parser.add_argument('--interleaved_split', action='store_true', default=True, help="Use an interleaved splitting strategy (default: True).")
    parser.add_argument('--train_chunk_size', type=int, default=8, help="For interleaved split, number of train samples in a cycle.")
    parser.add_argument('--validation_chunk_size', type=int, default=1, help="For interleaved split, number of validation samples in a cycle.")
    parser.add_argument('--test_chunk_size', type=int, default=1, help="For interleaved split, number of test samples in a cycle.")
    
    args = parser.parse_args()

    # --- Load Raw Dataset ---
    print(f"Loading raw dataset from '{args.dataset_path}'...")
    print(f"Dataset path: {os.path.abspath(args.dataset_path)}")

    dataset_path = os.path.abspath(args.dataset_path)

    dataset = load_dataset('json', data_files=dataset_path)['train']

    # --- Apply Prompt Formatting ---
    print("Applying prompt format...")
    processed_dataset = dataset.map(
        format_prompt,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count()
    )

    # --- Split Dataset (Pre-Tokenization) ---
    print("Splitting dataset using interleaved strategy...")
    
    train_chunk = args.train_chunk_size
    val_chunk = args.validation_chunk_size
    test_chunk = args.test_chunk_size
    cycle_len = train_chunk + val_chunk + test_chunk

    # Using filter with indices to create each split
    train_dataset = processed_dataset.filter(
        lambda _, i: i % cycle_len < train_chunk, 
        with_indices=True, num_proc=os.cpu_count()
    )
    validation_dataset = processed_dataset.filter(
        lambda _, i: train_chunk <= (i % cycle_len) < (train_chunk + val_chunk), 
        with_indices=True, num_proc=os.cpu_count()
    )
    test_dataset = processed_dataset.filter(
        lambda _, i: (train_chunk + val_chunk) <= (i % cycle_len), 
        with_indices=True, num_proc=os.cpu_count()
    )
    
    text_dataset_splits = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })
    
    print("Processed text dataset splits created:")
    for split, data in text_dataset_splits.items():
        print(f"- {split}: {len(data)} examples")
        
    # --- Save Processed Text-Only Dataset ---
    print(f"Saving processed text-only dataset to '{args.processed_output_dir}'...")
    os.makedirs(args.processed_output_dir, exist_ok=True)
    text_dataset_splits.save_to_disk(args.processed_output_dir)
    print("Text-only dataset saved successfully.")

    # --- Tokenize Dataset ---
    print(f"Loading tokenizer from '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    print("Tokenizing all splits...")
    tokenized_splits = text_dataset_splits.map(
        tokenize_function, 
        batched=True, 
        num_proc=os.cpu_count(),
        remove_columns=["text"] # Remove the raw text column after tokenization
    )
    
    print("Tokenized dataset splits created:")
    for split, data in tokenized_splits.items():
        print(f"- {split}: {len(data)} examples")
        
    # --- Save Final Tokenized Dataset ---
    print(f"Saving final tokenized dataset to '{args.tokenized_output_dir}'...")
    os.makedirs(args.tokenized_output_dir, exist_ok=True)
    tokenized_splits.save_to_disk(args.tokenized_output_dir)
    
    # Save the tokenizer with the dataset
    tokenizer.save_pretrained(args.tokenized_output_dir)
    print(f"Tokenizer also saved to '{args.tokenized_output_dir}'.")
    
    print("Data processing and tokenization complete.")

if __name__ == "__main__":
    main() 