
from datasets import load_dataset
import json

def main():
    dataset_name = "nvidia/OpenCodeInstruct"
    output_file = "output_dataset.jsonl"
    num_rows = 100

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=f"train[:{num_rows}]")

    print(f"Saving {len(dataset)} rows to {output_file}")
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print("Dataset successfully saved.")

if __name__ == "__main__":
    main() 