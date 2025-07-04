# Script to preprocess and tokenize data in a single pass
import os
import argparse
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer

def get_prompt(code_sample: str) -> str:
    """
    Formats the code sample into a prompt that encourages chain-of-thought reasoning.
    """
    instruction = "First, provide a step-by-step explanation of the Haskell code. Then, present the code itself within a formatted block."

    # During training, we provide the code, but we leave the explanation empty.
    # The model is trained on an "explanation -> code" format. It learns that a
    # complete response should have this structure. This helps preserve the
    # "reasoning" pathways in the model, as it learns to generate explanatory text
    # before the code.
    return f"""You are an expert Haskell programmer and a great teacher. Your task is to explain a Haskell function and then provide the code.

### Instruction:
{instruction}

### Response:
**Explanation:**
This Haskell code snippet is a function definition. Here is a breakdown of its purpose and functionality:

**Code:**
```haskell
{code_sample}
```"""

def main():
    parser = argparse.ArgumentParser(description="Prepare and tokenize Haskell dataset for SFT.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Hugging Face dataset name or local path.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the model for tokenization.")
    parser.add_argument('--output_dir', type=str, default='../data/sft_tokenized_haskell_dataset', help="Directory to save the processed and tokenized dataset.")
    parser.add_argument('--test_split_size', type=float, default=0.1, help="Fraction of the dataset to use for the test set.")
    parser.add_argument('--validation_split_size', type=float, default=0.1, help="Fraction of the training set to use for validation.")
    parser.add_argument('--max_length', type=int, default=4096, help="Max sequence length for tokenization.")
    
    args = parser.parse_args()

    # --- Load Raw Dataset ---
    print(f"Loading dataset from '{args.dataset_name}'...")
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name, split='train')

    # --- Apply Prompt Formatting ---
    print("Applying prompt format...")
    dataset = dataset.map(
        lambda example: {"text": get_prompt(example["code"])},
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count()
    )
    
    # --- Load Tokenizer ---
    print(f"Loading tokenizer from '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- Tokenize Dataset ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_length)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=os.cpu_count(), 
        remove_columns=["text"] # Remove the raw text column after tokenization
    )
    
    # --- Split Dataset ---
    print("Splitting dataset...")
    train_test_split = tokenized_dataset.train_test_split(test_size=args.test_split_size, shuffle=True, seed=42)
    train_val_split = train_test_split['train'].train_test_split(test_size=args.validation_split_size, shuffle=True, seed=42)
    
    processed_dataset = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': train_test_split['test']
    })
    
    print("Tokenized dataset splits created:")
    for split, data in processed_dataset.items():
        print(f"- {split}: {len(data)} examples")
        
    # --- Save Dataset ---
    print(f"Saving processed and tokenized dataset to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dataset.save_to_disk(args.output_dir)
    
    # Save the tokenizer with the dataset
    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer also saved to '{args.output_dir}'.")
    
    print("Data tokenization complete.")

if __name__ == "__main__":
    main() 