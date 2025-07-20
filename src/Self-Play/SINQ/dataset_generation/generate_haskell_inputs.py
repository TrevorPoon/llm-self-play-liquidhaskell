import os
import sys
import json
import logging
import argparse
import subprocess
import tempfile
import textwrap
import re                       # NEW
from typing import Optional     # NEW

from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset

# Adjust the path to import from utils and execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import get_function_name, get_function_arg_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Code Execution ------------------------------------------------------------
class CodeExecutor:
    """A wrapper for executing Haskell code."""
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.tmp_dir = tempfile.mkdtemp(prefix="haskell_executor_")

    def _create_haskell_program(self, program_code: str, main_body: str) -> str:
        """Builds a complete Haskell source file from a function definition and a main body."""
        prog = textwrap.dedent(program_code).strip()
        body = textwrap.dedent(main_body).rstrip()

        imports = """
import Prelude
import Data.List
import Data.Char
import Data.Maybe
import Text.Read (readMaybe)
""".strip()
        return f"{imports}\n\n{prog}\n\n{body}"

    def execute(self, program_code: str, input_str: str):
        """Executes a Haskell program with a given input and returns the stdout."""
        func_name = get_function_name(program_code)
        arg_type = get_function_arg_type(program_code)

        if not func_name or not arg_type:
            # logger.warning(f"Could not find function name or argument type in program. Assuming not executable. Program: {program_code}...")
            return "no_function", "Not executable"
        
        main_body = textwrap.dedent(f"""
main :: IO ()
main = do
    input <- getContents
    case readMaybe input :: Maybe {arg_type} of
      Just x  -> print ({func_name} x)
      Nothing -> error "Invalid input: expected {arg_type}"
        """).strip()

        program = self._create_haskell_program(program_code, main_body)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir) as f:
            f.write(program)
            f.flush()

            try:
                compile_process = subprocess.run(
                    ['/usr/bin/ghc', '-o', f.name.replace('.hs', ''), f.name], 
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                # logger.error(f"Compilation timed out for program: {program_code}... Input: {input_str}...")
                return "compile_timeout", "Compilation timed out"
            except Exception as e:
                # logger.error(f"Compilation error for program: {program_code}... Input: {input_str}... Error: {e}")
                return "compile_error", str(e)

            if compile_process.returncode != 0:
                # logger.error(f"Compilation failed for program: {program_code}... Input: {input_str}... Stderr: {compile_process.stderr}")
                return "compile_error", compile_process.stderr

            try:
                run_process = subprocess.run(
                    [f.name.replace('.hs', '')],
                    input=input_str,
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                # logger.error(f"Execution timed out for program: {program_code}... Input: {input_str}...")
                return "runtime_timeout", "Execution timed out"
            except Exception as e:
                # logger.error(f"Runtime error for program: {program_code}... Input: {input_str}... Error: {e}")
                return "runtime_error", str(e)

            if run_process.returncode != 0:
                # logger.error(f"Execution failed for program: {program_code}... Input: {input_str}... Stderr: {run_process.stderr}")
                return "runtime_error", run_process.stderr

            return "success", run_process.stdout.strip()

    def check_compiles_and_executes(self, program_code: str, input_str: str) -> bool:
        """Checks if a Haskell program compiles and executes successfully with the given input."""
        status, _ = self.execute(program_code, input_str)
        return status == "success"


# --- Input Generation ----------------------------------------------------------
class InputGenerator:
    def __init__(self, args):
        self.args = args
        self.executor = CodeExecutor(timeout=args.timeout)

    def load_programs(self, dataset_name):
        logger.info(f"Loading programs from dataset {dataset_name}...")
        programs = []
        try:
            # Load the dataset from the specified directory
            dataset = load_from_disk(dataset_name)
            # Extract 'code' from each item in the dataset
            for program_item in dataset:
                if 'code' in program_item and program_item['code']:
                    programs.append(program_item['code'])
        except Exception as e:
            logger.error(f"Failed to load dataset from disk {dataset_name}. Error: {e}")
            return []
        logger.info(f"Loaded {len(programs)} programs.")

        # programs = programs[:100]
        # logger.info(f"Filtered to {len(programs)} programs.")

        return programs

    # ---------------------------------------------------------------------
    # NEW, more general _generate_simple_input
    # ---------------------------------------------------------------------
    BASE_LITERALS = {
        "Int": "42",
        "Integer": "42",
        "Float": "3.14",
        "Double": "2.71",
        "Bool": "True",
        "Char": "'a'",
        "String": "\"hello\"",
        "()": "()"
    }
    DEFAULT_POLY = "Int"

    def _generate_simple_input(self, arg_type: str) -> Optional[str]:
        """Generate a small literal that Prelude.read will parse for arg_type."""
        # Reject clear non-parsable shapes
        if any(tok in arg_type for tok in ["->", "IO", "Monad"]):
            if "=>" not in arg_type:          # context constraints are fine
                return None

        # Drop context constraints like "(Num a) =>"
        arg_type = re.sub(r"^\s*\(.*?\)\s*=>\s*", "", arg_type).strip()

        # Replace all single-letter type vars with DEFAULT_POLY
        type_vars = sorted(set(re.findall(r"\b[a-z]\b", arg_type)))
        for v in type_vars:
            arg_type = re.sub(rf"\b{v}\b", self.DEFAULT_POLY, arg_type)

        # Recursive emitter ------------------------------------------------
        def emit(t: str) -> Optional[str]:
            t = t.strip()

            # List “[T]”
            if re.fullmatch(r"\[.*\]", t):
                inner = t[1:-1].strip() or "()"
                sample = emit(inner)
                return f"[{sample}, {sample}]" if sample else None

            # Maybe T
            if t.startswith("Maybe "):
                inner = t[len("Maybe "):]
                sample = emit(inner)
                return f"Just {sample}" if sample else "Nothing"

            # Either a b
            m = re.match(r"Either\s+(.+?)\s+(.+)$", t)
            if m:
                left = emit(m.group(1))
                right = emit(m.group(2))
                return f"Left {left}" if left else f"Right {right}"

            # Tuple "(T1, T2, …)"
            if t.startswith("(") and t.endswith(")"):
                elems = split_top_commas(t[1:-1])
                parts = [emit(e) for e in elems]
                if any(p is None for p in parts):
                    return None
                return f"({', '.join(parts)})"

            # Atomic
            return self.BASE_LITERALS.get(t)

        # Helper: split tuple contents at top-level commas
        def split_top_commas(s: str):
            depth = 0
            cur, out = [], []
            for ch in s:
                if ch == ',' and depth == 0:
                    out.append("".join(cur))
                    cur = []
                else:
                    if ch in "([<": depth += 1
                    if ch in ")]>": depth -= 1
                    cur.append(ch)
            out.append("".join(cur))
            return [p.strip() for p in out if p.strip()]

        return emit(arg_type)

    # ---------------------------------------------------------------------

    def generate_and_validate_inputs(self):
        programs = self.load_programs(self.args.dataset_name)
        
        output_jsonl_data = []
        error_counts = {
            "no_function": 0,
            "compile_timeout": 0,
            "compile_error": 0,
            "runtime_timeout": 0,
            "runtime_error": 0,
            "failed_generation": 0,
            "unknown_error": 0
        }

        # Prepare output directories
        os.makedirs(self.args.output_dir, exist_ok=True)
        output_jsonl_path = os.path.join(self.args.output_dir, "haskell_dataset_with_generated_inputs.jsonl")

        for i, program_code in enumerate(tqdm(programs, desc="Generating and Validating Inputs")):
            logger.info(f"Processing program {i+1}/{len(programs)}")

            func_name = get_function_name(program_code)
            arg_type = get_function_arg_type(program_code)

            if not func_name or not arg_type:
                # logger.warning(f"Could not find function name or argument type in program: {program_code}...")
                error_counts["no_function"] += 1
                continue

            input_str = self._generate_simple_input(arg_type)
            if input_str is None:
                # logger.warning(f"Failed to generate input for type {arg_type} in program: {program_code}...")
                error_counts["failed_generation"] += 1
                continue

            status, output = self.executor.execute(program_code, input_str)

            if status != "success":
                error_counts[status] += 1
            else:
                record = {
                    "code": program_code,
                    "input": input_str,
                    "status": status,
                    "output": output
                }
                output_jsonl_data.append(record)

        # Write to JSONL file
        with open(output_jsonl_path, 'w') as outfile:
            for record in output_jsonl_data:
                outfile.write(json.dumps(record) + '\n')
        logger.info(f"Output JSONL written to: {output_jsonl_path}")

        # Save as Hugging Face dataset
        if output_jsonl_data:
            hf_dataset = Dataset.from_list(output_jsonl_data)
            output_hf_dataset_path = self.args.output_hf_dataset_dir
            os.makedirs(output_hf_dataset_path, exist_ok=True)
            hf_dataset.save_to_disk(output_hf_dataset_path)
            logger.info(f"Output Hugging Face dataset saved to: {output_hf_dataset_path}")
        else:
            logger.warning("No successful generations to save to Hugging Face dataset.")

        logger.info("\n--- Summary of Input Generation and Validation ---")
        logger.info(f"Total programs processed: {len(programs)}")
        for k, v in error_counts.items():
            logger.info(f"{k.replace('_', ' ').title()}: {v}")


# --- CLI -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate and validate Haskell inputs.")
    parser.add_argument("--dataset_name", type=str, default="output_codes.jsonl",
                        help="Path to the JSONL file with Haskell programs.")
    parser.add_argument("--output_dir", type=str, default="./generated_inputs_output",
                        help="Directory to save the generated inputs and results.")
    parser.add_argument("--output_hf_dataset_dir", type=str, default="./generated_hf_dataset",
                        help="Directory to save the generated Hugging Face dataset (Arrow/Parquet format).")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Timeout (seconds) for Haskell compilation and execution.")

    args = parser.parse_args()

    generator = InputGenerator(args)
    generator.generate_and_validate_inputs()
    logger.info("Input generation and validation complete.")


if __name__ == "__main__":
    main()
