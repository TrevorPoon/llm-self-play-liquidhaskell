import json
import os
import shutil
import re

def transfer_tests():
    """
    Transfers the "test" field from humaneval-hs-ans.jsonl to
    humaneval-hs.jsonl based on matching "task_id".

    This version identifies and reports malformed lines without removing them.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ans_file_path = os.path.join(script_dir, 'humaneval-hs-ans.jsonl')
    main_file_path = os.path.join(script_dir, 'humaneval-hs.jsonl')
    temp_file_path = main_file_path + '.tmp'

    # 1. Read all tests from the answer file into a dictionary.
    tests_data = {}
    try:
        with open(ans_file_path, 'r', encoding='utf-8') as f_ans:
            for line in f_ans:
                if line.strip():
                    data = json.loads(line)
                    tests_data[data['task_id']] = data['test']
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {ans_file_path}: {e}")
        return

    # 2. Read the main file, update tests, and write to a temporary file.
    try:
        with open(main_file_path, 'r', encoding='utf-8') as f_main, \
             open(temp_file_path, 'w', encoding='utf-8') as f_temp:
            for line in f_main:
                if not line.strip():
                    f_temp.write(line)
                    continue
                
                try:
                    data = json.loads(line)
                    task_id = data.get('task_id')
                    
                    if task_id in tests_data:
                        data['test'] = tests_data[task_id]
                    
                    # Write the (potentially modified) object to the temp file.
                    f_temp.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    # Try to find task_id in the malformed line
                    match = re.search(r'"task_id":\s*"([^"]+)"', line)
                    if match:
                        task_id = match.group(1)
                        print(f"Malformed JSON found for task_id: {task_id}. Preserving original line.")
                    else:
                        print(f"Malformed JSON found (task_id not detectable). Preserving original line.")
                    
                    # Write the original malformed line to the temp file to preserve it.
                    f_temp.write(line)
    
    except IOError as e:
        print(f"Error processing {main_file_path}: {e}")
        # Clean up the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return

    # 3. Atomically replace the original file with the updated one.
    try:
        shutil.move(temp_file_path, main_file_path)
        print(f"Successfully transferred tests to {os.path.basename(main_file_path)}, preserving all original lines.")
    except (IOError, OSError) as e:
        print(f"Error replacing file: {e}")

if __name__ == '__main__':
    transfer_tests()
