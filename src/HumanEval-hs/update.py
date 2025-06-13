import json
import os
from pathlib import Path

def update_jsonl():
    """
    Updates the humaneval-hs.jsonl file with canonical solutions from .hs files
    and tests from haskell_humaneval.jsonl.
    """
    hs_solutions_path = Path("humaneval-hs")
    original_jsonl_path = Path("humaneval-hs.jsonl")
    new_tests_jsonl_path = Path("humaneval-hs-ans-final.jsonl")
    output_jsonl_path = Path("humaneval-hs-updated.jsonl")

    # 1. Load new tests from haskell_humaneval.jsonl into a dictionary
    new_tests = {}
    if new_tests_jsonl_path.exists():
        with open(new_tests_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    task_id = data['task_id']
                    test_content = data.get('test', '')
                    
                    # Remove lines containing 'import'
                    lines = test_content.split('\n')
                    filtered_lines = [l for l in lines if 'import' not in l]
                    cleaned_test_content = '\n'.join(filtered_lines)

                    new_tests[task_id] = cleaned_test_content
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num} of {new_tests_jsonl_path}: {e}")
                    print(f">>> {line.strip()}")
                except KeyError:
                    print(f"Warning: Skipping line {line_num} in {new_tests_jsonl_path} due to missing 'task_id' key.")
    else:
        print(f"Warning: '{new_tests_jsonl_path}' not found. 'test' fields will not be updated.")

    # 2. Load canonical solutions from the humaneval-hs directory
    solutions = {}
    if hs_solutions_path.is_dir():
        for hs_file in hs_solutions_path.glob("HumanEval-*.hs"):
            task_id_num = hs_file.stem.split('-')[1]
            task_id = f"Haskell/{task_id_num}"
            with open(hs_file, 'r', encoding='utf-8') as f:
                solutions[task_id] = f.read()
    else:
        print(f"Warning: Directory '{hs_solutions_path}' not found. 'canonical_solution' fields will not be updated.")

    # 3. Read original jsonl, update, and write to a new file
    if not original_jsonl_path.exists():
        print(f"Error: '{original_jsonl_path}' not found. Cannot perform update.")
        return

    updated_lines = []
    with open(original_jsonl_path, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                task_id = data['task_id']

                if task_id in solutions:
                    data['canonical_solution'] = solutions[task_id]
                else:
                    print(f"Warning: No solution found for {task_id}")

                if task_id in new_tests:
                    data['test'] = new_tests[task_id]
                else:
                    print(f"Warning: No test found for {task_id}")

                updated_lines.append(json.dumps(data))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num} of {original_jsonl_path}: {e}")
                print(f">>> {line.strip()}")
            except KeyError:
                print(f"Warning: Skipping line {line_num} in {original_jsonl_path} due to missing 'task_id' key.")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(updated_lines) + '\n')

    print(f"Successfully created updated file at: '{output_jsonl_path}'")
    
    # To replace the original file, you can uncomment these lines:
    # os.remove(original_jsonl_path)
    # os.rename(output_jsonl_path, original_jsonl_path)
    # print(f"Original file '{original_jsonl_path}' has been replaced with the updated version.")


if __name__ == "__main__":
    update_jsonl()
