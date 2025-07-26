# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import random
import logging
import argparse
import subprocess
import shutil
from tqdm import tqdm
from peft import PeftModel
import vllm
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import textwrap
import tempfile
from datasets import load_dataset, load_from_disk, Dataset
from typing import List
import re

from utils.utils import get_function_name, get_function_arg_type, print_nvidia_smi, print_gpu_memory_usage, strip_comments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Prompts (from SInQ Paper, Appendix C) ---

ALICE_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Haskell programmer. Your task is to generate a semantically inequivalent variant of a given Haskell program, which means that there must exist at least a diverging input example such that the original program and your program either produce different outputs or exceptions, or one halts and the other one does not halt.
    You must also provide a diverging input, which is a valid input for both programs, but on which they produce different outputs.
                                      
    A good inequivalent program `Q` should be subtly different from `P`.
    A good diverging input `x` should be simple and clearly demonstrate the semantic difference between `P` and `Q`.

    The original program and your program will be used in a test to evaluate the skill of an expert Haskell programmer who will have to produce a diverging example (not necessarily the same as yours), so make sure that the difference you introduce are not very easy to understand. 
    You will be given a difficulty level from 0 (easiest) to 10 (hardest) to target. E.g. difficulty level 0 means that an expert computer scientist in the bottom decile or above should be able to find a diverging example, difficulty level 9 means that only an expert computer scientist in the top decile should be able to find a diverging example, and difficulty level 10 means that only the top 0.01 or less of expert Haskell programmer should be able to find a diverging example.                                 

    First, think step-by-step and write down your analysis of program `P` and your strategy for creating an inequivalent program `Q`. Enclose this reasoning within `<think>` and `</think>` tags.
    After the thinking block, the final answer could **only** be in the following format, without any additional explanation or context.

    **Generated Program `Q`:**
    ```haskell
    <Your generated Haskell code for `Q`>
    ```

    **Diverging Input `x`:**

    First, think step-by-step and write down your analysis of program `P` and your strategy for creating an inequivalent program `Q`. Enclose this reasoning within `<think>` and `</think>` tags.
    After the thinking block, the final answer could **only** be in the following format, without any additional explanation or context.

    Final output MUST be exactly: 
    **Generated Program `Q`:**
    ```haskell
    <Your generated Haskell code for `Q`>
    ```

    **Diverging Input `x`:**
    ```
    <The diverging input `x`>
    ```

""").strip()

ALICE_USER_PROMPT = textwrap.dedent("""
    Difficulty level: {difficulty_level}
    Original program `P`:
    ```haskell
    {program}
    ```

    <think>
""").strip()

BOB_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Haskell programmer. You are given two Haskell programs, `P` and `Q`.
    Your task is to determine if they are semantically equivalent.
    If they are inequivalent, you must provide a diverging input `x` on which `P(x) != Q(x)`.

    If the programs are equivalent, respond with "The programs are equivalent."
    If they are inequivalent, respond with your thought process and the diverging input could **only** be in the following markdown format, without any additional explanation or context:

    **Analysis:**
    <Your analysis of the differences between `P` and `Q`.>

    **Diverging Input `x`:**
    ```
    <The diverging input `x`>
    ```                                
                                    
""").strip()

BOB_USER_PROMPT_TEMPLATE = textwrap.dedent("""
    Program `P`:
    ```haskell
    {program_p}
    ```

    Program `Q`:
    ```haskell
    {program_q}
    ```

    <think>
""")

ALICE_DIFFICULTY_PREDICTION_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
    Difficulty level: Any
    ```haskell
    {program}
    ```
""").strip()

ALICE_DIFFICULTY_PREDICTION_USER_PROMPT = textwrap.dedent("""
    Predict the difficulty level of the instance. Just write \"Difficulty level: D\" where D is your prediction, do not write anything else.
""").strip()

# --- Code Execution ---

class CodeExecutor:
    """A wrapper for executing Haskell code and comparing results."""
    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="haskell_executor_")

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

    def execute(self, program_code: str, input: any):
        """Executes a Haskell program with a given input and returns the stdout."""
        func_name = get_function_name(program_code)
        arg_type = get_function_arg_type(program_code)

        logger.info(f"Function name: {func_name}")
        logger.info(f"Argument type: {arg_type}")
        logger.info(f"Input: {input}")

        if not func_name or not arg_type:
            logger.warning(f"Could not find function name or argument type in program. Assuming not executable.")
            return "no_function", "Not executable"
        
        main_body = textwrap.dedent(f"""
main :: IO ()
main = do
    line <- getLine
    case readMaybe line :: Maybe {arg_type} of
        Just arg -> print ({func_name} arg)
        Nothing -> error "Invalid input: expected {arg_type}"
        """).strip()

        program = self._create_haskell_program(program_code, main_body)

        logger.info(f"Program: \n\n{program}\n\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir.name) as f:
            f.write(program)
            f.flush()

            try:
                compile_process = subprocess.run(
                    ['/usr/bin/ghc', '-o', f.name.replace('.hs', ''), f.name], 
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning("Compilation timed out.")
                return "compile_timeout", "Compilation timed out"
            except Exception as e:
                logger.warning(f"An unexpected error occurred during compilation: {e}")
                return "compile_error", str(e)

            if compile_process.returncode != 0:
                return "compile_error", compile_process.stderr

            try:
                run_process = subprocess.run(
                    [f.name.replace('.hs', '')],
                    input=input,
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning("Execution timed out.")
                return "runtime_timeout", "Execution timed out"
            except Exception as e:
                logger.warning(f"An unexpected error occurred during execution: {e}")
                return "runtime_error", str(e)

            if run_process.returncode != 0:
                return "runtime_error", run_process.stderr

            return "success", run_process.stdout.strip()

    def check_divergence(self, p: str, q: str, x: any) -> bool:
        """Checks if two programs diverge on a given input."""
        if p == q:
            logger.warning(f"Programs are the same. Returning False.")
            return False
        
        status_p, out_p = self.execute(p, x)
        status_q, out_q = self.execute(q, x)

        logger.info(f"Status P: {status_p}")
        logger.info(f"Status Q: {status_q}")
        logger.info(f"Output P: {out_p}")
        logger.info(f"Output Q: {out_q}")


        # If both succeeded but outputs are different, they diverge
        if status_p == "success" and status_q == "success" and out_p != out_q:
            return True

        if status_p != status_q:
            return True

        return False


    def check_compiles(self, program_code: str) -> bool:
        """Checks if a Haskell program compiles successfully."""
        main_body = 'main :: IO ()\nmain = putStrLn "compiles"'
        program = self._create_haskell_program(program_code, main_body)

        logger.info(f"Program: \n\n{program}\n\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir.name) as f:
            f.write(program)
            f.flush()
            try:
                compile_process = subprocess.run(
                    ['/usr/bin/ghc', '-c', f.name],
                    capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning("GHC type-check timed out.")
                return False
            except Exception as e:
                logger.warning(f"An unexpected error occurred during GHC type-check: {e}")
                return False

            if compile_process.returncode != 0:
                logger.warning(f"GHC type-check error:\n\n{compile_process.stderr}")
                return False
        return True


# --- SInQ Self-Play ---
class SInQ:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name_or_path
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        torch.cuda.empty_cache()
        self._initialize_vllm()

        self.alice_adapter_path = args.alice_adapter_path
        self.bob_adapter_path = args.bob_adapter_path

        self.executor = CodeExecutor(timeout=args.timeout)
        self.programs = self.load_programs(args.dataset_name)

        self.cumulative_alice_training_data = self.load_cumulative_training_data(args.cumulative_alice_training_data_path)
        self.cumulative_bob_training_data = self.load_cumulative_training_data(args.cumulative_bob_training_data_path)


    def _initialize_vllm(self):
        logger.info("Initializing vLLM model...")
        self.vllm_model = vllm.LLM(
            model=self.model_name,
            tensor_parallel_size=self.args.tensor_parallel_size,
            trust_remote_code=True,
            enable_lora=True,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_tokens,
        )

        # print_gpu_memory_usage("After initializing vLLM model")
        logger.info("After initializing vLLM model (nvidia-smi):")
        print_nvidia_smi(f"After initializing vLLM model")
        logger.info("After initializing vLLM model (GPU memory usage):")
        print_gpu_memory_usage(f"After initializing vLLM model")

    def load_cumulative_training_data(self, programs_file_path):     
        if programs_file_path and os.path.exists(programs_file_path):
            logger.info(f"Loading programs from {programs_file_path}...")
            with open(programs_file_path, 'r') as f:
                return [json.loads(line.strip()) for line in f if line.strip()]
        return []

    def load_programs(self, dataset_name):
        
        logger.info(f"Loading initial programs from dataset {dataset_name}...")
        try:
            programs = []
            dataset_iterator = None

            if os.path.isdir(dataset_name):
                logger.info(f"Found local dataset directory. Loading from disk: {dataset_name}")
                dataset = load_from_disk(dataset_name)
                # Get the top N programs, ensuring not to go out of bounds
                num_to_take = self.args.num_initial_programs
                if num_to_take==0:
                    logger.info("`num_initial_programs` is not set, loading all programs.")
                    dataset_iterator = dataset
                else:
                    num_to_take = min(num_to_take, len(dataset))
                    logger.info(f"Initial programs: {len(dataset)}")
                    logger.info(f"Loading {num_to_take} initial programs.")
                    dataset_iterator = dataset.select(range(num_to_take))
            else:
                logger.info(f"Loading from Hugging Face Hub: {dataset_name}")
                dataset = load_dataset(dataset_name, split='train', streaming=True)
                if self.args.num_initial_programs==0:
                    logger.info("`num_initial_programs` is not set, loading all programs from stream.")
                    dataset_iterator = dataset
                else:
                    logger.info(f"Loading {self.args.num_initial_programs} initial programs from stream.")
                    dataset_iterator = dataset.take(self.args.num_initial_programs)

            for program_item in dataset_iterator:
                if 'code' in program_item and program_item['code']:
                    programs.append(program_item['code'])

            if self.args.num_initial_programs!=0 and len(programs) < self.args.num_initial_programs:
                 logger.warning(f"Warning: Only able to load {len(programs)} programs, but {self.args.num_initial_programs} were requested.")

            logger.info(f"Loaded {len(programs)} programs from {dataset_name}.")
            return programs

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}. If using a local path, have you run the preprocessing script? Error: {e}")
            # Exit or return an empty list if dataset loading fails
            return []

    def parse_alice_output(self, text):
        try:
            # logger.info(f"Alice output: \n\n{text}\n\n")
            # Extract program Q
            program_q_match = re.search(r"\*\*Generated Program `Q`:\*\*\s*```haskell\n(.*?)\n```", text, re.DOTALL)
            if not program_q_match:
                logger.error("🟥 --- Alice parsing failed: Could not find 'Generated Program Q' block ---")

                logger.error(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            program_q = program_q_match.group(1).strip()

            # Extract diverging input x
            diverging_input_match = re.search(r"\*\*Diverging Input `x`:\*\*\s*```(?:[^\n]*)\n(.*?)\n```", text, re.DOTALL)
            if not diverging_input_match:
                logger.error("🟥 --- Alice parsing failed: Could not find 'Diverging Input x' block ---")
                logger.error(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            diverging_input = diverging_input_match.group(1).strip()
            # Remove potential 'x = ' prefix from the diverging input.
            diverging_input = re.sub(r'^\s*\w+\s*=\s*', '', diverging_input).strip()
            
            if not program_q or not diverging_input:
                logger.error("🟥 --- Alice parsing failed: `Q` or `x` is empty after parsing ---")
                logger.error(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            
            return program_q, diverging_input
        except Exception as e:
            logger.error(f"🟥 --- Alice parsing failed with exception: {e} ---")
            logger.error(f"Full output from Alice:\n{text}")
            return None, None

    def parse_bob_output(self, text):
        try:
            # Extract diverging input
            diverging_input_match = re.search(r"\*\*Diverging Input `x`:\*\*\s*```(?:[^\n]*)\n(.*?)\n```", text, re.DOTALL)
            if diverging_input_match:
                diverging_input = diverging_input_match.group(1).strip()
                # Remove potential 'x = ' prefix from the diverging input.
                diverging_input = re.sub(r'^\s*\w+\s*=\s*', '', diverging_input).strip()
                return diverging_input
            return None
        except Exception as e:
            logger.error(f"🟥 --- Bob parsing failed with exception: {e} ---")
            return None

    def run_alice(self, program_p):
        """Alice generates a variant of a program."""
        logger.info(f"Running Alice...")
        
        user_content = ALICE_USER_PROMPT.format(
            difficulty_level=10,
            program=program_p
        )
        messages = [
            {"role": "system", "content": ALICE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
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
        
        # Note: vLLM currently doesn't support different LoRA adapters for different requests in a single batch.
        # We pass the LoRA request, and it will be applied to all prompts in the batch.
        
        for attempt in range(self.args.max_alice_retries):
            logger.info(f"Alice generation attempt {attempt + 1}/{self.args.max_alice_retries}...")
            request_outputs = self.vllm_model.generate(
                [prompt],
                sampling_params,
                lora_request=LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
            )

            logger.info(f"Length of Alice request_outputs: {len(request_outputs)}")

            # Assuming n=1 for Alice, so only one output to process
            output = request_outputs[0]
            result_text = output.outputs[0].text
            program_q, diverging_input = self.parse_alice_output(result_text)

            if program_q and diverging_input:
                # Before returning, check if the generated Q compiles
                if self.executor.check_compiles(program_q):
                    logger.info(f"Alice successfully generated a compilable program Q on attempt {attempt + 1}.")
                    return program_q, diverging_input, result_text
                else:
                    logger.warning(f"🟨 Alice generated a program Q that failed to compile on attempt {attempt + 1}. Retrying...")
            else:
                logger.warning(f"🟨 Alice parsing failed on attempt {attempt + 1}. Retrying...")
        
        logger.error(f"🟥 Alice failed to generate a compilable program after {self.args.max_alice_retries} attempts.")
        return None, None, None

    def run_bob(self, program_p, program_q):
        """
        Bob checks for semantic equivalence and calculates difficulty.
        It generates n_samples and checks how many are correct.
        """
        logger.info(f"Running Bob to calculate difficulty over {self.args.n_samples} attempts...")
        
        messages = [
            {"role": "system", "content": BOB_SYSTEM_PROMPT},
            {"role": "user", "content": BOB_USER_PROMPT_TEMPLATE.format(program_p=program_p, program_q=program_q)}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = vllm.SamplingParams(
            n=self.args.n_samples,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_tokens,
            presence_penalty=self.args.presence_penalty
        )
        
        # We generate n_samples for a single prompt. This returns one RequestOutput.
        request_outputs = self.vllm_model.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("bob_adapter", 1, self.bob_adapter_path) if self.bob_adapter_path else None
        )

        logger.info(f"Length of Bob request_outputs: {len(request_outputs)}")
        
        n_correct = 0
        valid_diverging_inputs = []
        bob_training_responses = []

        # The single RequestOutput contains a list of n_samples CompletionOutputs
        bob_outputs = request_outputs[0].outputs
        for output in bob_outputs:
            result_text = output.text
            diverging_input = self.parse_bob_output(result_text)

            logger.info(f"Bob's diverging input: {diverging_input}")
            
            if diverging_input:
                if self.executor.check_divergence(program_p, program_q, diverging_input):
                    n_correct += 1
                    valid_diverging_inputs.append(diverging_input)
                    # Add the raw successful response for Bob's training
                    bob_training_responses.append(result_text)

        difficulty = 10 * (1 - (n_correct / self.args.n_samples))
        logger.info(f"✅ Bob found {n_correct}/{self.args.n_samples} correct diverging inputs. Calculated difficulty: {difficulty:.2f}")
        
        return difficulty, valid_diverging_inputs, bob_training_responses

    def save_as_hf_jsonl(self, data: List[dict], file_path: str):
        # Detect if data is list of str, and wrap it
        if data and isinstance(data[0], str):
            records = [{"text": line} for line in data]
        else:
            records = data  # assume it's already List[dict]

        # 'data' is a list of dicts
        ds = Dataset.from_list(records)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ds.to_json(file_path, orient="records", lines=True)
        logger.info(f"Saved HF JSONL dataset with {len(ds)} examples to {file_path}")

    def run_generation_iteration(self):
        logger.info(f"--- Starting Data Generation Iteration {self.args.iteration} ---")
            
        run_bob_triggered_count = 0
        
        warning_counts = {
            "p_compile_fail": 0,
            "alice_generation_fail": 0,
            "q_compile_fail": 0,
            "alice_not_divergent": 0,
        }
        
        candidate_examples = []
        
        # random.shuffle(self.programs)
        
        for j, p_original in enumerate(tqdm(self.programs, desc=f"Iteration {self.args.iteration}")):
            tqdm.write(f"\n--- Processing example {j+1}/{len(self.programs)} in Iteration {self.args.iteration} ---")

            p_original = p_original.replace("main :: IO ()", "")
            p_original = strip_comments(p_original)
            
            tqdm.write(f"Original program (P):\n{p_original}")

            if not self.executor.check_compiles(p_original):
                logger.warning("🟨 Original program P failed to compile. Skipping example.")
                warning_counts["p_compile_fail"] += 1
                continue

            # Alice's turn
            q_candidate, x_candidate, alice_raw_output = self.run_alice(p_original)
            
            if not q_candidate:
                logger.warning("🟨 Alice failed to generate a candidate program.")
                warning_counts["alice_generation_fail"] += 1
                continue

            if not self.executor.check_compiles(q_candidate):
                logger.warning("🟨 Candidate Q failed to compile. Skipping example.")
                warning_counts["q_compile_fail"] += 1
                continue
        
            logger.info(f"Alice's candidate program: \n{q_candidate}")
            logger.info(f"Alice's diverging input: {x_candidate}")

            is_divergent_alice = self.executor.check_divergence(p_original, q_candidate, x_candidate)
            
            if not is_divergent_alice:
                logger.warning("🟨 Alice's candidate program and diverging input were not divergent / Execution failed. Skipping.")
                warning_counts["alice_not_divergent"] += 1
                continue

            # Bob's turn: calculate difficulty and get training data
            run_bob_triggered_count += 1
            difficulty, valid_bob_inputs, bob_successful_responses = self.run_bob(p_original, q_candidate)

            # Store candidate for Alice's training data, to be filtered later
            candidate_examples.append({
                "p_original": p_original,
                "q_candidate": q_candidate,
                "alice_raw_output": alice_raw_output,
                "difficulty": difficulty
            })

            # Create Bob's training data from his successful attempts
            for bob_response in bob_successful_responses:
                # Convert to desired format: 'system prompt', 'user prompt', 'output'
                bob_training_example = {
                    "system_prompt": BOB_SYSTEM_PROMPT,
                    "user_prompt": BOB_USER_PROMPT_TEMPLATE.format(program_p=p_original, program_q=q_candidate),
                    "output": bob_response
                }
                self.cumulative_bob_training_data.append(bob_training_example)

            # If Bob completely failed, but Alice provided a valid input, create a hard training example for Bob
            if not valid_bob_inputs:
                logger.info("🟨 Bob failed to find any diverging input. Creating a hard training example for Bob from Alice's input.")
                bob_completion = f"\n**Analysis:**\n<analysis>\n\n**Diverging Input `x`:**\n```\n{x_candidate}\n```"
                # Convert to desired format: 'system prompt', 'user prompt', 'output'
                bob_training_example = {
                    "system_prompt": BOB_SYSTEM_PROMPT,
                    "user_prompt": BOB_USER_PROMPT_TEMPLATE.format(program_p=p_original, program_q=q_candidate),
                    "output": bob_completion
                }
                self.cumulative_bob_training_data.append(bob_training_example)
        
        # --- Task 3: Biasing Alice's Training Set ---
        hard_examples = [ex for ex in candidate_examples if ex["difficulty"] >= self.args.difficulty_threshold]
        easy_examples = [ex for ex in candidate_examples if ex["difficulty"] < self.args.difficulty_threshold]

        # Group easy examples by integer difficulty bins
        num_easy_bins = int(self.args.difficulty_threshold) if self.args.difficulty_threshold >= 1 else 1
        bins = {k: [] for k in range(num_easy_bins)}
        for ex in easy_examples:
            bin_idx = int(ex["difficulty"])
            if bin_idx < num_easy_bins:
                bins[bin_idx].append(ex)

        # Sample 20% of the number of hard examples from the easy examples
        needed = int(0.2 * len(hard_examples))
        sampled_easies = []
        
        # Round-robin sampling from bins
        if easy_examples:
            i = 0
            while len(sampled_easies) < needed:
                bin_idx = i % num_easy_bins
                if bins[bin_idx]:
                    sampled_easies.append(bins[bin_idx].pop(random.randrange(len(bins[bin_idx]))))
                i += 1
                # Break if we have cycled through all bins and can't find any more examples
                if i > 0 and i % num_easy_bins == 0:
                    if sum(len(b) for b in bins.values()) == 0:
                        break
        
        final_alice_examples = hard_examples + sampled_easies

        for ex in final_alice_examples:
            # --- (A) Main SFT Example ---
            main_sft_user_content = ALICE_USER_PROMPT.format(
                difficulty_level=f"{ex['difficulty']:.2f}",
                program=ex['p_original']
            )
            # Convert to desired format: 'system prompt', 'user prompt', 'output'
            alice_training_example = {
                "system_prompt": ALICE_SYSTEM_PROMPT,
                "user_prompt": main_sft_user_content,
                "output": ex['alice_raw_output']
            }
            self.cumulative_alice_training_data.append(alice_training_example)

            # --- (B) Difficulty-Prediction Example ---
            diff_pred_user_content = ALICE_DIFFICULTY_PREDICTION_SYSTEM_PROMPT_TEMPLATE.format(
                program=ex['p_original'],
            )
            # Convert to desired format: 'system prompt', 'user prompt', 'output'
            diff_pred_training_example = {
                "system_prompt": "", # No system prompt for this one, as per original structure
                "user_prompt": diff_pred_user_content + ex['alice_raw_output'] + ALICE_DIFFICULTY_PREDICTION_USER_PROMPT,
                "output": f"Difficulty level: {ex['difficulty']:.2f}"
            }
            self.cumulative_alice_training_data.append(diff_pred_training_example)
        
        logger.info(f"Iteration {self.args.iteration} summary:")
        logger.info(f"  - Bob was triggered {run_bob_triggered_count} times.")
        logger.info(f"  - Total candidates generated: {len(candidate_examples)}")
        logger.info(f"  - Hard examples (d >= {self.args.difficulty_threshold}): {len(hard_examples)}")
        logger.info(f"  - Easy examples (d < {self.args.difficulty_threshold}): {len(easy_examples)}")
        logger.info(f"  - Sampled easy examples: {len(sampled_easies)}")
        logger.info(f"  - Alice training data size this iteration: {len(self.cumulative_alice_training_data)}")
        logger.info(f"  - Average difficulty of hard examples: {sum([ex['difficulty'] for ex in hard_examples]) / len(hard_examples) if hard_examples else 0:.2f}")   
        logger.info(f"  - Average difficulty of easy examples: {sum([ex['difficulty'] for ex in easy_examples]) / len(easy_examples) if easy_examples else 0:.2f}")
        logger.info(f"  - Average difficulty of all examples: {sum([ex['difficulty'] for ex in candidate_examples]) / len(candidate_examples) if candidate_examples else 0:.2f}")
        logger.info(f"  - Bob training data size: {len(self.cumulative_bob_training_data)}")
        logger.info("  - 🟥 🟥 🟥 Warning counts: 🟥 🟥 🟥")
        logger.info(f"    - Original program P failed to compile: {warning_counts['p_compile_fail']}")
        logger.info(f"    - Alice failed to generate a candidate program: {warning_counts['alice_generation_fail']}")
        logger.info(f"    - Candidate Q failed to compile: {warning_counts['q_compile_fail']}")
        logger.info(f"    - Alice's candidate not divergent / Execution failed: {warning_counts['alice_not_divergent']}")

        # --- Save artifacts ---
        iter_output_dir = self.args.iteration_dir
        
        # Save training data
        if self.cumulative_alice_training_data:
            self.save_as_hf_jsonl(self.cumulative_alice_training_data, os.path.join(iter_output_dir, "alice_training_data.jsonl"))
        if self.cumulative_bob_training_data:
            self.save_as_hf_jsonl(self.cumulative_bob_training_data, os.path.join(iter_output_dir, "bob_training_data.jsonl"))
        
        self.save_as_hf_jsonl([warning_counts], os.path.join(iter_output_dir, "warning_counts.jsonl"))
        self.save_as_hf_jsonl(candidate_examples, os.path.join(iter_output_dir, "candidate_examples.jsonl"))

        logger.info("Generation iteration complete.")


def main():
    parser = argparse.ArgumentParser(description="SInQ Data Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--dataset_name', type=str, default='../data/successfully_compiled_sorted_haskell_dataset', help="Hugging Face dataset name or local path for initial programs.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save models and results.")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens for the model's context window and generation.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for sampling.")
    parser.add_argument('--top_k', type=int, default=20, help="Top-k for sampling.")
    parser.add_argument('--presence_penalty', type=float, default=1.5, help="Presence penalty for sampling.")
    parser.add_argument('--timeout', type=float, default=20.0, help="Timeout for code execution.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples for Bob to generate.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--num_initial_programs', type=int, default=0, help="Number of initial programs to load.")
    parser.add_argument('--difficulty_threshold', type=float, default=3.0, help="Difficulty threshold for filtering Alice's training data.")
    parser.add_argument('--alice_adapter_path', type=str, default=None, help="Path to an initial LoRA adapter for Alice.")
    parser.add_argument('--bob_adapter_path', type=str, default=None, help="Path to an initial LoRA adapter for Bob.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of tensor parallel processes.")
    parser.add_argument('--max_alice_retries', type=int, default=5, help="Maximum number of retries for Alice to generate a compilable program.")
    parser.add_argument('--iteration', type=int, required=True, help="Current self-play iteration number.")
    parser.add_argument('--cumulative_alice_training_data_path', type=str, default=None, help="Path to a file with the list of programs for the current iteration.")
    parser.add_argument('--cumulative_bob_training_data_path', type=str, default=None, help="Path to a file with the list of programs for the current iteration.")
    parser.add_argument('--iteration_dir', type=str, default=None, help="Path to the directory for the current iteration.")
    
    args = parser.parse_args()

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    sinq = SInQ(args)
    sinq.run_generation_iteration()

if __name__ == "__main__":
    main()



