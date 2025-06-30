import argparse
from datasets import load_dataset
import os

def main():
    parser = argparse.ArgumentParser(description="Preprocess and sort Haskell code dataset from Hugging Face.")
    parser.add_argument('--dataset_name', type=str, default='blastwind/github-code-haskell-file', help="Hugging Face dataset name.")
    parser.add_argument('--output_dir', type=str, default='../../data/sorted_haskell_dataset', help="Directory to save the sorted dataset.")
    args = parser.parse_args()

    # Check if output directory exists.
    output_path = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading dataset {args.dataset_name}...")
    # Load the full dataset, not streaming, to be able to sort it.
    dataset = load_dataset(args.dataset_name, split='train')
    
    print(f"Sorting dataset by 'size' ascendingly...")
    # The sort function in `datasets` is efficient. It returns a new Dataset object.
    sorted_dataset = dataset.sort("size")
    
    
    print(f"Saving sorted dataset to {output_path}...")
    # save_to_disk saves the dataset in Arrow format.
    sorted_dataset.save_to_disk(output_path)

    print("Preprocessing finished successfully.")
    print(f"You can now use the path '{args.output_dir}' in the --dataset_name argument of run_blastwind.py")

if __name__ == "__main__":
    main() 