import json
from datasets import load_from_disk

def read_and_extract_code(num_rows=1000):
    output_data = []
    # Load the dataset from Hugging Face
    dataset = load_from_disk("../../data/SINQ_sorted_blastwind_haskell_dataset")

    # Iterate through the dataset and extract code
    for i, data in enumerate(dataset):
        # if i >= num_rows:
        #     break
        if 'code' in data:
            output_data.append(json.dumps({'code': data['code']}))
    return output_data

if __name__ == "__main__":
    extracted_codes = read_and_extract_code(num_rows=100000000)
    output_filename = "output_codes_blastwind.jsonl"
    with open(output_filename, 'w') as outfile:
        for code_jsonl in extracted_codes:
            outfile.write(code_jsonl + '\n')
