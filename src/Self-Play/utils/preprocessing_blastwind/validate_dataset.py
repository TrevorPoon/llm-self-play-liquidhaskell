# -*- coding: utf-8 -*-
import os
import sys
import logging
import argparse
import subprocess
import textwrap
import tempfile
import re
import json
from datasets import load_from_disk, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_function_name(program_code: str):
    """Extracts function name from a Haskell code snippet."""
    match = re.search(r"^([\w']+)\s*::", program_code, re.MULTILINE)
    if match:
        return match.group(1)
    return None

def has_function_binding(program_code: str) -> bool:
    """
    Checks if the Haskell code snippet has a function binding for the declared function.
    """
    # Check for an equals sign, which is a strong indicator of a binding.
    # This is a simple but effective heuristic for this problem.
    if '=' not in program_code:
        return False

    # Also check if the function name is used in a binding.
    function_name = get_function_name(program_code)
    if function_name:
        # Pattern to find `func ... = `
        # This is not perfect but will catch most simple function bindings.
        # It looks for the function name at the start of a line (with optional parens for operators)
        # and an equals sign later on that line.
        pattern = re.compile(f"^{re.escape(function_name)}\\s*.*=", re.MULTILINE)
        if re.search(pattern, program_code):
            return True
        # if the above fails, it might be a more complex binding pattern that our regex doesn't catch
        # but the simple '=' check is a good fallback.
    
    # As a fallback, if we found an '=' but no function name or complex binding,
    # we assume it's valid for now. The compiler will be the final check.
    return True

def has_input_arguments(program_code: str) -> bool:
    """
    Checks if a Haskell function has at least one input argument in its type signature.
    A function is considered to have input arguments if its type signature line
    contains '->'. It ignores comments.
    """
    for line in program_code.split('\n'):
        # remove comments
        line_without_comments = line.split('--')[0]
        if '::' in line_without_comments and '->' in line_without_comments:
            return True
    return False

class CodeExecutor:
    """A wrapper for executing Haskell code and comparing results."""
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.tmp_dir = tempfile.mkdtemp(prefix="haskell_executor_")

    def _create_haskell_program(self, program_code: str, main_body: str) -> str:
        """Builds a complete Haskell source file from a function definition and a main body."""
        prog = textwrap.dedent(program_code).strip()
        body = textwrap.dedent(main_body).rstrip()

        imports = """
import Data.List
import Data.Char
import Data.Maybe
import Text.Read (readMaybe)
        """.strip()

        return f"{imports}\n\n{prog}\n\n{body}"

    def check_compiles(self, program_code: str) -> bool:
        """Checks if a Haskell program compiles successfully."""
        main_body = 'main :: IO ()\nmain = putStrLn "compiles"'
        
        # We need a function name to create a valid main body for executables, 
        # but for a simple compile check, we don't need to execute anything.
        # We can just check if the code itself is valid.
        program = self._create_haskell_program(program_code, main_body)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir) as f:
            f.write(program)
            f.flush()
            compile_process = subprocess.run(
                ['ghc', '-v', '-c', f.name],
                capture_output=True, text=True, timeout=self.timeout
            )
            if compile_process.returncode != 0:
                # We can optionally log the error for debugging
                logger.warning(f"GHC type-check error:\n\n{compile_process.stderr}")
                return False
            else:
                print(f"GHC type-check success:\n\n{compile_process.stdout}")
        return True

    def __del__(self):
        # Cleanup the temporary directory
        try:
            import shutil
            shutil.rmtree(self.tmp_dir)
        except OSError as e:
            # This can happen if the directory is already gone
            pass


def validate_dataset(args):
    """
    Loads a dataset, checks each program for compilation, and saves the valid ones to a new dataset.
    """
    logger.info(f"Loading dataset from {args.dataset_path}")
    try:
        dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}. Error: {e}")
        return
    
    # select the first 100 programs
    # dataset = dataset.select(range(1000))

    executor = CodeExecutor(timeout=args.timeout)
    
    valid_programs = []
    total_programs = len(dataset)
    programs_with_inputs = 0
    programs_with_binding = 0
    
    logger.info("Starting validation for all programs in the dataset...")
    for item in tqdm(dataset, desc="Validating programs"):
        code = item.get('code')
        if not code:
            continue
            
        if not has_input_arguments(code):
            continue
        
        programs_with_inputs += 1

        if not has_function_binding(code):
            continue

        programs_with_binding += 1

        if executor.check_compiles(code):
            valid_programs.append(item)
    
    logger.info("Validation finished.")
    logger.info(f"Original dataset size: {total_programs}")
    logger.info(f"Programs with input arguments: {programs_with_inputs}")
    logger.info(f"Programs with function binding: {programs_with_binding}")
    logger.info(f"Successfully compiled programs with inputs: {len(valid_programs)}")
    
    if valid_programs:
        output_dir = os.path.join(args.output_dir, "successfully_compiled_sorted_haskell_dataset")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a new Hugging Face Dataset from the list of dictionaries
        new_dataset = Dataset.from_list(valid_programs)
        
        logger.info(f"Saving new dataset with {len(valid_programs)} programs to {output_dir}")
        new_dataset.save_to_disk(output_dir)
        logger.info("Successfully saved the new dataset.")

        # Also save as a .jsonl file for easier inspection
        jsonl_output_path = os.path.join(args.output_dir, "validated_haskell_dataset.jsonl")
        logger.info(f"Saving validated dataset as a .jsonl file to {jsonl_output_path}")
        try:
            with open(jsonl_output_path, 'w', encoding='utf-8') as f:
                for program in new_dataset:
                    f.write(json.dumps(program) + '\n')
            logger.info("Successfully saved .jsonl file.")
        except IOError as e:
            logger.error(f"Failed to write to {jsonl_output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Validate Haskell programs in a dataset.")
    
    # Get the parent directory of the script's location to construct default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))

    default_dataset_path = os.path.join(parent_dir, 'data', 'sorted_haskell_dataset')
    default_output_dir = os.path.join(parent_dir, 'data')

    parser.add_argument('--dataset_path', type=str, default=default_dataset_path,
                        help='Path to the Hugging Face dataset on disk.')
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                        help='Directory to save the new dataset.')
    parser.add_argument('--timeout', type=float, default=20.0,
                        help='Timeout for the compilation check.')

    args = parser.parse_args()
    
    validate_dataset(args)

if __name__ == "__main__":
    main() 