import argparse
from datasets import load_from_disk
import os

def read_top_n(dataset_path, n=100):
    """
    Reads the top n rows from a dataset saved to disk.
    
    Args:
        dataset_path (str): The path to the dataset directory.
        n (int): The number of rows to read.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        print("Please make sure you have run the preprocessing script first.")
        return

    try:
        # Use load_from_disk for datasets saved with save_to_disk.
        print(f"Loading dataset from '{dataset_path}'...")
        dataset = load_from_disk(dataset_path)
        
        print(f"Reading top {n} rows...")
        
        # Slicing is efficient on memory-mapped datasets.
        # Ensure we don't request more rows than exist.
        num_rows_to_read = min(n, len(dataset))
        
        for i, example in enumerate(dataset.select(range(num_rows_to_read))):
            print(f"--- Record {i + 1}/{num_rows_to_read} ---")
            print(f"Size: {example.get('size', 'N/A')}")
            print(f"Code:\n{example.get('code', 'N/A')}")
            print("-" * 20)
            
        print(f"\nFinished reading {num_rows_to_read} rows.")

    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Default path assumes the script is run from the root of the project directory
    default_path = '../data/sorted_haskell_dataset'
    
    parser = argparse.ArgumentParser(description="Read top N rows from the sorted Haskell dataset.")
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default=default_path,
        help=f"Path to the sorted dataset directory. Defaults to '{default_path}'"
    )
    parser.add_argument(
        '--n', 
        type=int, 
        default=100,
        help="Number of rows to read and print."
    )
    args = parser.parse_args()

    read_top_n(args.dataset_path, args.n)

if __name__ == "__main__":
    main() 