import json

def read_and_extract_code(filepath, num_rows=1000):
    output_data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_rows:
                break
            try:
                data = json.loads(line)
                if 'code' in data:
                    output_data.append(json.dumps({'code': data['code']}))
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line.strip()}")
    return output_data

if __name__ == "__main__":
    file_path = "../../data/SINQ_compiled_sorted_blastwind_haskell_dataset_with_input/validated_haskell_dataset.jsonl"
    extracted_codes = read_and_extract_code(file_path, num_rows=100000000)
    output_filename = "output_codes.jsonl"
    with open(output_filename, 'w') as outfile:
        for code_jsonl in extracted_codes:
            outfile.write(code_jsonl + '\n')
