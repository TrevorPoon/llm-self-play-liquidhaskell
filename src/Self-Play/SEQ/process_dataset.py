import json
import subprocess
import os
import tempfile
import shutil
import re
import argparse
from tqdm import tqdm

def process_haskell_code(haskell_code_content):
    temp_dir = tempfile.mkdtemp()
    haskell_file_name = "MyTest.hs"
    haskell_file_path = os.path.join(temp_dir, haskell_file_name)

    try:
        has_module_decl = haskell_code_content.strip().startswith("module")
        
        final_haskell_code = haskell_code_content
        if not has_module_decl:
            header = """
module MyTempModule where

import Prelude
import Data.List
import Data.Char
import Data.Maybe
import Text.Read (readMaybe)
"""
            final_haskell_code = header.strip() + "\n\n" + haskell_code_content

            print(f"Final Haskell code: {final_haskell_code}")

        with open(haskell_file_path, "w") as f:
            f.write(final_haskell_code)

        liquid_command = ["ghc", "-fplugin=LiquidHaskell", haskell_file_name]

        env = os.environ.copy()
        env['PATH'] = f"{os.path.expanduser('~')}/.local/bin:{env['PATH']}"

        process = subprocess.run(
            liquid_command,
            check=False,
            capture_output=True,
            text=True,
            cwd=temp_dir,
            env=env
        )

        full_output = process.stdout + process.stderr
        
        print(f"Liquid command output: {full_output}")
        print("="*100)

        liquid_summary = "LIQUID_SUMMARY_NOT_FOUND"
        for line in process.stdout.splitlines():
            if "LIQUID:" in line:
                liquid_summary = line.strip()
                break
    

        if "LIQUID: SAFE" in liquid_summary:
            return "LIQUID:SAFE", liquid_summary, full_output
        elif "LIQUID: UNSAFE" in liquid_summary:
            return "liquid_unsafe_error", liquid_summary, full_output
        elif process.returncode != 0:
            return "compilation_error", liquid_summary, full_output
        else:
            return "unknown_error", liquid_summary, full_output

    except Exception as e:
        return "execution_error", f"An unexpected error occurred: {e}", ""
    finally:
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Run LiquidHaskell check on completions.")
    parser.add_argument("--row", type=int, default=None, help="Specify a single row index to run.")
    parser.add_argument("--file", type=str, default="../data/synthetic_liquid_haskell_dataset_nvidia/synthetic_liquid_haskell_dataset_nvidia.jsonl", help="Path to dataset file")
    parser.add_argument("--output_file", type=str, default="safe_completions.jsonl", help="Path to output JSONL file for LIQUID:SAFE completions.")
    args = parser.parse_args()

    dataset_path = args.file
    single_row = args.row
    output_path = args.output_file

    output_file = None
    if output_path:
        output_file = open(output_path, 'w')

    stats = {
        "total_completions": 0,
        "code_extraction_failure": 0,
        "compilation_error": 0,
        "liquid_unsafe_error": 0,
        "LIQUID:SAFE": 0,
        "execution_error": 0,
    }

    print(f"Processing dataset from: {dataset_path}")

    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    if single_row is not None:
        if 0 <= single_row < len(lines):
            lines = [lines[single_row]]
        else:
            print(f"Row index {single_row} is out of range. Dataset has {len(lines)} rows.")
            return
        
    lines = lines[:100]

    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Processing completions"):
        stats["total_completions"] += 1
        data = json.loads(line)
        completion = data.get('completion', '')

        match = re.findall(r'```haskell\n(.*?)\n```', completion, re.DOTALL)
        haskell_code = match[-1].strip() if match else None

        if not haskell_code:
            stats["code_extraction_failure"] += 1
            print(f"⚠️ Code block not found for row {idx}")
            continue

        status, liquid_summary, full_output = process_haskell_code(haskell_code)

        if status == "compilation_error":
            stats["compilation_error"] += 1
            print(f"🟨 🟥 Compilation Error for row {idx}")
        elif status == "liquid_unsafe_error":
            stats["liquid_unsafe_error"] += 1
            print(f"🟨 🟥 Liquid Output Error for row {idx}")
        elif status == "LIQUID:SAFE":
            stats["LIQUID:SAFE"] += 1
            print(f"🟨 🟢 LIQUID:SAFE for row {idx}")
            if output_file:
                output_file.write(json.dumps(data) + '\n')
        elif status == "execution_error":
            stats["execution_error"] += 1
            print(f"🟨 🟥 Execution Error for row {idx}")
        else:
            print(f"Unexpected status: {status} for row {idx}")

    print("\n--- Summary ---")
    print(f"Total Completions Processed: {stats['total_completions']}")
    print(f"Code Extraction Failures: {stats['code_extraction_failure']}")
    print(f"Compilation Errors: {stats['compilation_error']}")
    print(f"Liquid Output Errors (UNSAFE/Not Found): {stats['liquid_unsafe_error']}")
    print(f"LIQUID:SAFE Checks: {stats['LIQUID:SAFE']}")
    print(f"Unexpected Execution Errors: {stats['execution_error']}")

    if output_file:
        output_file.close()

if __name__ == "__main__":
    main()
