import os
import sys
import json
import torch
import random
import logging
import argparse
import subprocess
import shutil
import tempfile
import textwrap
import re
import vllm
from transformers import AutoTokenizer


from tqdm import tqdm
from datasets import load_dataset, load_from_disk

# Adjust the path to import from utils and execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import get_function_name, get_function_arg_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Prompts ---
INPUT_GENERATION_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Haskell programmer. Your task is to generate a simple, valid input `x` for a given Haskell program.
    The input should be concise and demonstrate a typical use case of the function.

    First, think step-by-step and analyze the program to understand its expected input type and behavior. Enclose this reasoning within `<think>` and `</think>` tags.
    After the thinking block, the final answer could **only** be in the following format, without any additional explanation or context.

    **Generated Input `x`:**
    ```
    <Your generated input `x`>
    ```
""").strip()

INPUT_GENERATION_USER_PROMPT = textwrap.dedent("""
    Haskell program `P`:
    ```haskell
    {program}
    ```
""").strip()

# --- Code Execution ---
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
            logger.warning(f"Could not find function name or argument type in program. Assuming not executable. Program: {program_code}...")
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
                logger.error(f"Compilation timed out for program: {program_code}... Input: {input_str}...")
                return "compile_timeout", "Compilation timed out"
            except Exception as e:
                logger.error(f"Compilation error for program: {program_code}... Input: {input_str}... Error: {e}")
                return "compile_error", str(e)

            print(f"Compile process: {compile_process}")
            print(f"Compile process returncode: {compile_process.returncode}")
            print(f"Compile process stderr: {compile_process.stderr}")
            print(f"Compile process stdout: {compile_process.stdout}")

            if compile_process.returncode != 0:
                logger.error(f"Compilation failed for program: {program_code}... Input: {input_str}... Stderr: {compile_process.stderr}")
                return "compile_error", compile_process.stderr

            try:
                run_process = subprocess.run(
                    [f.name.replace('.hs', '')],
                    input=input_str,
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.error(f"Execution timed out for program: {program_code}... Input: {input_str}...")
                return "runtime_timeout", "Execution timed out"
            except Exception as e:
                logger.error(f"Runtime error for program: {program_code}... Input: {input_str}... Error: {e}")
                return "runtime_error", str(e)

            if run_process.returncode != 0:
                logger.error(f"Execution failed for program: {program_code}... Input: {input_str}... Stderr: {run_process.stderr}")
                return "runtime_error", run_process.stderr

            return "success", run_process.stdout.strip()

    def check_compiles_and_executes(self, program_code: str, input_str: str) -> bool:
        """Checks if a Haskell program compiles and executes successfully with the given input."""
        status, _ = self.execute(program_code, input_str)
        return status == "success"

class InputGenerator:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._initialize_vllm()
        self.executor = CodeExecutor(timeout=args.timeout)

    def _initialize_vllm(self):
        logger.info("Initializing vLLM model...")
        self.vllm_model = vllm.LLM(
            model=self.model_name,
            tensor_parallel_size=self.args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_tokens,
        )

    def load_programs(self, dataset_name):
        logger.info(f"Loading programs from dataset {dataset_name}...")
        programs = []
        try:
            if os.path.isdir(dataset_name):
                dataset = load_from_disk(dataset_name)
            else:
                dataset = load_dataset(dataset_name, split='train', streaming=True)
            
            for program_item in tqdm(dataset, desc="Loading programs"):
                if 'code' in program_item and program_item['code']:
                    programs.append(program_item['code'])
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}. Error: {e}")
            return []
        logger.info(f"Loaded {len(programs)} programs.")

        programs = programs[:100]
        logger.info(f"Filtered to {len(programs)} programs.")
        
        return programs

    def parse_llm_output(self, text):
        try:
            m = re.search(
                r"\*\*Generated Input `x`\*\*:\s*```[^\n]*\n([\s\S]*?)```",
                text
            )
            return m.group(1).strip() if m else None
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")
            return None

    def generate_and_validate_inputs(self):
        programs = self.load_programs(self.args.dataset_name)
        
        output_dataset = []
        error_counts = {
            "no_function": 0,
            "compile_timeout": 0,
            "compile_error": 0,
            "runtime_timeout": 0,
            "runtime_error": 0,
            "failed_generation": 0,
            "unknown_error": 0
        }

        output_file_path = os.path.join(self.args.output_dir, "haskell_dataset_with_validated_inputs.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)

        for i, program_code in enumerate(tqdm(programs, desc="Generating and Validating Inputs")):
            logger.info(f"Processing program {i+1}/{len(programs)}")
            
            messages = [
                {"role": "system", "content": INPUT_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": INPUT_GENERATION_USER_PROMPT.format(program=program_code)}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            sampling_params = vllm.SamplingParams(
                n=1,
                temperature=self.args.temperature,
                max_tokens=self.args.max_tokens,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                presence_penalty=self.args.presence_penalty
            )
            
            try:
                request_outputs = self.vllm_model.generate([prompt], sampling_params)
                generated_input = self.parse_llm_output(request_outputs[0].outputs[0].text)

                print(f"Program code: {program_code}")
                print(f"Generated input: {generated_input}")

                if generated_input:
                    logger.info(f"Generated input: {generated_input}")
                    if self.executor.check_compiles_and_executes(program_code, generated_input):
                        logger.info("Program compiled and executed successfully with generated input.")
                        output_dataset.append({
                            "program": program_code,
                            "generated_input": generated_input,
                            "status": "passed"
                        })
                        with open(output_file_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps({
                                "program": program_code,
                                "generated_input": generated_input,
                                "status": "passed"
                            }) + '\n')
                    else:
                        status, _ = self.executor.execute(program_code, generated_input) # Re-execute to get the exact status
                        logger.warning(f"Program failed to compile or execute with generated input. Status: {status}")
                        error_counts[status] += 1
                        output_dataset.append({
                            "program": program_code,
                            "generated_input": generated_input,
                            "status": status
                        })
                else:
                    logger.warning("Failed to generate a valid input from LLM.")
                    output_dataset.append({
                        "program": program_code,
                        "generated_input": None,
                        "status": "failed_generation"
                    })
                    error_counts["failed_generation"] += 1
            except Exception as e:
                logger.error(f"Error during LLM generation or execution check for program {i+1}: {e}")
                output_dataset.append({
                    "program": program_code,
                    "generated_input": None,
                    "status": f"error: {e}"
                })
                error_counts["unknown_error"] += 1

        logger.info(f"Processed {len(programs)} programs. Successfully saved {len([p for p in output_dataset if p['status'] == 'passed'])} valid examples to {output_file_path}")
        logger.info("Error Summary:")
        for error_type, count in error_counts.items():
            logger.info(f"  {error_type}: {count}")
        return output_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate and Validate Inputs for Haskell Programs")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--dataset_name', type=str, default='../data/SINQ_compiled_sorted_blastwind_haskell_dataset_with_input', help="Hugging Face dataset name or local path for initial programs.")
    parser.add_argument('--output_dir', type=str, default='generated_inputs_output', help="Directory to save the generated dataset.")
    parser.add_argument('--max_tokens', type=int, default=1024, help="Maximum number of tokens for the model's generation.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument('--top_k', type=int, default=50, help="Top-k for sampling.")
    parser.add_argument('--presence_penalty', type=float, default=1.0, help="Presence penalty for sampling.")
    parser.add_argument('--timeout', type=float, default=30.0, help="Timeout for code execution.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of tensor parallel processes for vLLM.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="The fraction of GPU memory to be used for the vLLM KV cache.")

    args = parser.parse_args()

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    print(torch.version.cuda)
    
    generator = InputGenerator(args)
    generator.generate_and_validate_inputs()

if __name__ == "__main__":
    main() 