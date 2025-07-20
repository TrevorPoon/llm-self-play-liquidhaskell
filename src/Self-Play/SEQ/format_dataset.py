import json
import os
import re
import ast

input_file_path = 'output_dataset.jsonl'
output_dir = 'datasets'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create the 'nvidia' subdirectory if it doesn't exist
nvidia_output_dir = os.path.join(output_dir, 'nvidia')
os.makedirs(nvidia_output_dir, exist_ok=True)

file_counter = 0

with open(input_file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        file_counter += 1
        
        task_id = data['id']
        raw_output = data['output']
        raw_unit_tests = data['unit_tests']

        # Process output
        # Remove markdown fences and strip leading/trailing whitespace
        clean_output = raw_output.replace("```python", "").replace("```", "").strip()

        # Use AST to determine if it's a function or class and extract the main definition
        function_signature = ""
        function_name = ""
        is_class = False
        full_definition_code = ""

        try:
            tree = ast.parse(clean_output)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    # Find the exact lines of the function definition
                    lines = clean_output.splitlines()
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    full_definition_code = "\n".join(lines[start_line:end_line])
                    
                    # Extract the signature from the first line of the definition
                    sig_match = re.match(r"(def\s+\w+\(.*\):)", lines[start_line])
                    function_signature = sig_match.group(1) if sig_match else ""
                    break
                elif isinstance(node, ast.ClassDef):
                    function_name = node.name # Use function_name for class name for consistency
                    is_class = True
                    # Find the exact lines of the class definition
                    lines = clean_output.splitlines()
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    full_definition_code = "\n".join(lines[start_line:end_line])
                    
                    # For classes, the "signature" is just the class declaration line
                    sig_match = re.match(r"(class\s+\w+.*:)", lines[start_line])
                    function_signature = sig_match.group(1) if sig_match else ""
                    break
        except SyntaxError:
            # Fallback if the code isn't valid Python, treat as a single block
            full_definition_code = clean_output
            function_name = "" # Cannot reliably determine name
            function_signature = ""
            is_class = False

        # Find docstring within the full_definition_code
        docstring_match = re.search(r'''"""(.|\n)*?"""''', full_definition_code)
        docstring = docstring_match.group(0) if docstring_match else ""

        # Check for typing import
        typing_import = ""
        if ':' in full_definition_code or '->' in full_definition_code: # Check for types anywhere in the definition
            typing_import = "from typing import *\n\n"

        # Construct function part (now general for function or class)
        # The full_definition_code already contains the body and correct indentation
        function_part = full_definition_code

        # Process unit tests
        tests_list = ast.literal_eval(raw_unit_tests)
        formatted_tests = []
        for test_str in tests_list:
            # Replace original function/class name with 'candidate'
            if function_name:
                if is_class:
                    # For classes, replace ClassName() with candidate and ClassName.method() with candidate.method()
                    test_str = re.sub(r'\b' + re.escape(function_name) + r'\(\)((?:\.\w+)?)', r'candidate\1', test_str)
                    test_str = re.sub(r'\b' + re.escape(function_name) + r'\.', 'candidate.', test_str)
                else:
                    # For functions, replace functionName( with candidate(
                    test_str = re.sub(r'\b' + re.escape(function_name) + r'\(', 'candidate(', test_str)
            
            # Indent each line of the test
            indented_test = "\n".join(["    " + line.strip() for line in test_str.strip().split('\n')])
            formatted_tests.append(indented_test)
        
        # Assemble the final script
        check_candidate_line = ""
        if is_class:
            check_candidate_line = f"    check({function_name}())" # Instantiate the class
        else:
            check_candidate_line = f"    check({function_name})" # Pass the function directly

        final_script = f"""{typing_import}{function_part}

### Unit tests below ###
def check(candidate):
{"\n".join(formatted_tests)}

def test_check():
{check_candidate_line}
"""
        # Save to file
        new_output_file_path = os.path.join(nvidia_output_dir, f"OpenCodeInstruct_{file_counter}_{function_name}.py")
        with open(new_output_file_path, 'w') as out_f:
            out_f.write(final_script)

print(f"Processed {file_counter} tasks and saved to '{nvidia_output_dir}' directory.") 