import json
import subprocess
import os
import tempfile
import shutil
import re

def process_haskell_code(haskell_code_content):
    """
    Processes a single Liquid Haskell code block by writing it to a temporary file,
    running Liquid Haskell, and capturing the output.
    Returns a tuple (success_status, liquid_summary, full_output).
    """
    temp_dir = tempfile.mkdtemp()
    haskell_file_name = "MyTest.hs"
    haskell_file_path = os.path.join(temp_dir, haskell_file_name)

    try:
        with open(haskell_file_path, "w") as f:
            f.write(haskell_code_content)

        liquid_command = ["ghc", "-fplugin=LiquidHaskell", haskell_file_name]

        env = os.environ.copy()
        env['PATH'] = f"{os.path.expanduser('~')}/.local/bin:{env['PATH']}"

        process = subprocess.run(
            liquid_command,
            check=False,  # Do not raise CalledProcessError, we want to capture stdout/stderr
            capture_output=True,
            text=True,
            cwd=temp_dir,
            env=env
        )

        print(f"Process: {process}")
        print(f"Process.stdout: {process.stdout}")
        print(f"Process.stderr: {process.stderr}")

        full_output = process.stdout + process.stderr
        liquid_summary = "LIQUID_SUMMARY_NOT_FOUND"
        for line in process.stdout.splitlines():
            if "LIQUID:" in line:
                liquid_summary = line.strip()
                break
        
        if process.returncode != 0:
            return "compilation_error", liquid_summary, full_output
        elif "LIQUID: SAFE" in liquid_summary:
            return "LIQUID:SAFE", liquid_summary, full_output
        elif "LIQUID: UNSAFE" in liquid_summary or "LIQUID_SUMMARY_NOT_FOUND" in liquid_summary:
            return "liquid_output_error", liquid_summary, full_output
        else: # Catch any other unexpected success from ghc
            return "unknown_error", liquid_summary, full_output

    except Exception as e:
        return "execution_error", f"An unexpected error occurred: {e}", ""
    finally:
        shutil.rmtree(temp_dir)

def main():
    dataset_path = "../data/synthetic_liquid_haskell_dataset/synthetic_liquid_haskell_dataset.jsonl"

    stats = {
        "total_completions": 0,
        "code_extraction_failure": 0,
        "compilation_error": 0,
        "liquid_output_error": 0, # Includes LIQUID: UNSAFE and LIQUID_SUMMARY_NOT_FOUND
        "LIQUID:SAFE": 0,
        "execution_error": 0, # For unexpected Python errors during processing
    }

    print(f"Processing dataset from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        for line in f:
            stats["total_completions"] += 1
            data = json.loads(line)
            completion = data.get('completion', '')

            # Regex to find the last ```haskell ... ``` block
            # This regex looks for the last occurrence of ```haskell, then any characters (non-greedy)
            # until the next ```. The re.DOTALL flag is important for matching across newlines.
            match = re.findall(r'```haskell\n(.*?)\n```', completion, re.DOTALL)
            
            haskell_code = None
            if match:
                haskell_code = match[-1].strip() # Get the last match and strip whitespace

            print(f"Haskell Code: {haskell_code}")

            if not haskell_code:
                stats["code_extraction_failure"] += 1
                continue

            status, liquid_summary, full_output = process_haskell_code(haskell_code)
            
            if status == "compilation_error":
                stats["compilation_error"] += 1
            elif status == "liquid_output_error":
                stats["liquid_output_error"] += 1
            elif status == "LIQUID:SAFE":
                stats["LIQUID:SAFE"] += 1
            elif status == "execution_error":
                stats["execution_error"] += 1
            else:
                # This should ideally not be hit if all cases are covered
                print(f"Unexpected status: {status} for completion: {completion[:100]}...")
    
    print("\n--- Summary ---")
    print(f"Total Completions Processed: {stats['total_completions']}")
    print(f"Code Extraction Failures: {stats['code_extraction_failure']}")
    print(f"Compilation Errors: {stats['compilation_error']}")
    print(f"Liquid Output Errors (UNSAFE/Not Found): {stats['liquid_output_error']}")
    print(f"LIQUID:SAFE Checks: {stats['LIQUID:SAFE']}")
    print(f"Unexpected Execution Errors: {stats['execution_error']}")

if __name__ == "__main__":
    main() 