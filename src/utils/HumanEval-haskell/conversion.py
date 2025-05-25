import os
import json
import re

def parse_doctests(haskell_code_lines):
    """
    Parses Haskell doctest examples (lines starting with -- >>>)
    and the expected output lines immediately following.
    Returns a list of (expression, expected_output) tuples.
    """
    doctests = []
    i = 0
    while i < len(haskell_code_lines):
        line = haskell_code_lines[i].strip()
        if line.startswith("-- >>>"):
            expression = line.replace("-- >>>", "").strip()
            # Expect result on the next line(s) starting with --
            i += 1
            expected_output_lines = []
            while i < len(haskell_code_lines) and haskell_code_lines[i].strip().startswith("-- ") and not haskell_code_lines[i].strip().startswith("-- >>>"):
                expected_output_lines.append(haskell_code_lines[i].strip().replace("-- ", "", 1))
                i += 1
            if expected_output_lines:
                doctests.append((expression, "\n".join(expected_output_lines)))
            # if no expected output line, it might be an expression that doesn't return or error (less common for humaneval)
            # or it's the end of parsing for this doctest block
            continue # continue to next line with outer loop
        i += 1
    return doctests

def generate_haskell_test_main(task_id, function_name, doctests):
    if not doctests:
        return ""

    test_prints = []
    for i, (expression, expected) in enumerate(doctests):
        # Basic handling for multiline expected output: compare as string literals
        # More complex types might need more sophisticated parsing/comparison
        if '\n' in expected:
            # Represent multiline strings in Haskell test
            # This is a simplified approach; direct comparison of complex multiline outputs
            # might be tricky without a proper test framework.
            # We'll rely on Haskell's Show instance for now.
            pass # Keep as is, rely on Haskell's Show for lists/tuples of strings

        # Ensure generated expression to be tested uses the actual function name from the signature
        # The `expression` from doctest might use a slightly different name or casing.
        # This is a heuristic: replace the first word (assumed to be function name)
        # in the doctest expression with the extracted function_name
        parts = expression.split(" ", 1)
        if len(parts) > 0 and function_name:
             # Check if the first part of the expression is likely the function name to be replaced
             # This is a simple heuristic. More robust parsing might be needed if func names are complex
             if parts[0].isalnum() : # Simple check if it's an identifier
                 expression_to_test = function_name + (" " + parts[1] if len(parts) > 1 else "")
             else: # If not a simple identifier (e.g. operator section), use original
                 expression_to_test = expression

        else:
            expression_to_test = expression
        
        test_prints.append(f"    putStrLn $俱乐部 ((show ({expression_to_test})) == (show ({expected})))")


    haskell_test_script = f"""module Main where

-- Solution will be prepended here by the evaluation script
-- {task_id}

main :: IO ()
main = do
{chr(10).join(test_prints)}

-- Helper to avoid issues with string escaping in f-string and then in Haskell
俱乐部 :: String -> String
俱乐部 s = s
"""
    return haskell_test_script.replace("俱乐部", "{-CLS-}") # Placeholder to be replaced after f-string formatting


def parse_hs_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    task_id_match = re.search(r"-- Task ID: (HumanEval/\d+)", content)
    # Default to a generic task_id if not found, using the filename
    filename_task_id = "Haskell/" + os.path.basename(filepath).replace("HumanEval-","").replace(".hs","")
    task_id = task_id_match.group(1).replace("HumanEval", "Haskell") if task_id_match else filename_task_id


    prompt_parts = []
    signature = ""
    solution_lines = []
    
    lines = content.split('\n')
    in_haskell_impl = False
    found_signature = False
    haskell_code_lines_for_doctest = [] # Store lines relevant for doctest parsing

    for line in lines:
        if "Haskell Implementation:" in line:
            in_haskell_impl = True
            continue

        if not in_haskell_impl:
            continue
        
        haskell_code_lines_for_doctest.append(line) # Collect all lines after "Haskell Implementation"

        # Current prompt extraction logic
        if line.strip().startswith("-- >>>") or (line.strip().startswith("-- ") and not line.strip().startswith("-- Task ID:")):
            prompt_parts.append(line.strip()) # Keep -- for prompt
        elif "::" in line and not found_signature :
            signature = line.strip()
            prompt_parts.append(signature)
            found_signature = True
        elif found_signature and line.strip() and not line.strip().startswith("--"):
            solution_lines.append(line)
        elif found_signature and solution_lines and (not line.strip() or line.strip().startswith("--")): # End of solution
            break 
            
    prompt = '\n'.join(prompt_parts)
    canonical_solution = '\n'.join(solution_lines).strip()

    # Extract function name from signature for test generation
    function_name = ""
    if signature:
        match = re.match(r"\s*(\w+)\s*::", signature)
        if match:
            function_name = match.group(1)

    # Parse doctests from the collected Haskell lines
    parsed_doctests = parse_doctests(haskell_code_lines_for_doctest)
    test_code = generate_haskell_test_main(task_id, function_name, parsed_doctests)
    test_code = test_code.replace("{-CLS-}", "") # Remove placeholder

    return {
        "task_id": task_id,
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "test": test_code,
    }

def convert_hs_to_jsonl(hs_dir, output_file):
    with open(output_file, 'w') as outfile:
        # Iterate from 0 to 163 (inclusive) for HumanEval task IDs
        for i in range(164): 
            hs_filename = f"HumanEval-{i}.hs"
            hs_filepath = os.path.join(hs_dir, hs_filename)
            if os.path.exists(hs_filepath):
                data = parse_hs_file(hs_filepath)
                # Ensure task_id is consistently "Haskell/i" for the output jsonl
                data["task_id"] = f"Haskell/{i}" 
                outfile.write(json.dumps(data) + '\n')
            else:
                print(f"Warning: File {hs_filepath} not found.")

if __name__ == "__main__":
    # Assuming the script is in llm-self-play-liquidhaskell/src/utils/HumanEval-haskell/
    hs_dataset_dir = "./humaneval-hs" 
    # Output to HumanEval/data directory, relative to the script's parent's parent.
    output_jsonl_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "HumanEval", "data", "humaneval-hs.jsonl"
    )
    
    os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)
    
    convert_hs_to_jsonl(hs_dataset_dir, output_jsonl_file)
    print(f"Conversion complete. Output written to {output_jsonl_file}")
