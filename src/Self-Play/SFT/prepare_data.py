# Script to preprocess and tokenize data
import os
import argparse
from datasets import load_dataset, load_from_disk, DatasetDict

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
    parser = argparse.ArgumentParser(description="Prepare Haskell dataset for SFT.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Hugging Face dataset name or local path.")
    parser.add_argument('--output_dir', type=str, default='../data/sft_processed_haskell_dataset', help="Directory to save the processed dataset.")
    parser.add_argument('--test_split_size', type=float, default=0.1, help="Fraction of the dataset to use for the test set.")
    parser.add_argument('--validation_split_size', type=float, default=0.1, help="Fraction of the training set to use for validation.")

    args = parser.parse_args()

    print(f"Loading dataset from '{args.dataset_name}'...")
    if os.path.isdir(args.dataset_name):
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name, split='train')

    # Apply prompt formatting and remove original columns in one step
    dataset = dataset.map(
        lambda example: {"text": get_prompt(example["code"])},
        remove_columns=dataset.column_names
    )
    
    # Split into train/test
    train_test_split = dataset.train_test_split(test_size=args.test_split_size, shuffle=True, seed=42)
    
    # Split train into train/validation
    train_val_split = train_test_split['train'].train_test_split(test_size=args.validation_split_size, shuffle=True, seed=42)
    
    processed_dataset = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': train_test_split['test']
    })
    
    print("Dataset splits created:")
    for split, data in processed_dataset.items():
        print(f"- {split}: {len(data)} examples")
        
    print(f"Saving processed dataset to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dataset.save_to_disk(args.output_dir)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    main() 