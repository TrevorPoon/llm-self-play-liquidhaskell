import json
import os

script_dir = os.path.dirname(__file__)
input_file = os.path.join(script_dir, 'humaneval-hs.jsonl')
output_file = os.path.join(script_dir, 'extracted_tests.txt')

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        task_id = data.get('task_id')
        test_content = data.get('test')

        if task_id and test_content:
            f_out.write(f"Task ID: {task_id}\n")
            f_out.write("Test:\n")
            f_out.write(test_content.replace('\\n', '\n') + "\n")
            f_out.write("-" * 30 + "\n") 