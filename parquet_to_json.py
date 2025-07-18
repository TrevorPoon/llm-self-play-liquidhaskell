import pandas as pd
import json
import argparse
import numpy as np

def parquet_to_jsonl(parquet_file, jsonl_file):
    """
    Converts a Parquet file to a JSONL file.
    Each row in the Parquet file will be written as a single JSON object per line in the JSONL file.
    """
    try:
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(parquet_file)

        # Open the output JSONL file in write mode
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            # Iterate over DataFrame rows and write each as a JSON object
            for i, row in df.iterrows():
                json_record = row.to_dict()
                # Add task_id
                json_record["task_id"] = f"MBPP/{i + 1}"
                # Convert any numpy arrays or pandas Series in the dictionary to lists
                for key, value in json_record.items():
                    if isinstance(value, np.ndarray) or isinstance(value, pd.Series):
                        json_record[key] = value.tolist()
                f.write(json.dumps(json_record) + '\n')
        
        print(f"Successfully converted '{parquet_file}' to '{jsonl_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{parquet_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Parquet file to a JSONL file.")
    parser.add_argument("parquet_input", help="Path to the input Parquet file.")
    parser.add_argument("jsonl_output", help="Path to the output JSONL file.")
    
    args = parser.parse_args()

    # Define the input and output file paths
    # Note: The user specified an absolute path for the input parquet file in the prompt
    input_parquet_path = args.parquet_input
    output_jsonl_path = args.jsonl_output

    parquet_to_jsonl(input_parquet_path, output_jsonl_path)
