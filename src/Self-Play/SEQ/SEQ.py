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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import vllm
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import textwrap
import tempfile
from datasets import load_dataset, load_from_disk

from torch.utils.data import Dataset
import re

from utils.utils import extract_generation_code, get_function_name, get_function_arg_type, print_nvidia_smi, print_gpu_memory_usage, strip_comments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Prompts (from SInQ Paper, Appendix C) ---

ALICE_SYSTEM_PROMPT = textwrap.dedent("""
    You are a helpful and expert Haskell programmer, powered by Liquid Haskell.
    Your job is to *transform* any given function `P` into a new function `Q` that:

      • Is syntactically correct Haskell.  
      • Is semantically equivalent: ∀x. `P x == Q x`.  
      • Uses a *different* implementation and a different function name (e.g. add a trailing `'`).  

    Always think through your transformation steps in `<think>…</think>`, then emit exactly:

    **Generated Program `Q`:**
    ```haskell
    <your Q here>
    ```
""").strip()


ALICE_USER_PROMPT = textwrap.dedent("""
    Here is the original Haskell function `P`:

    ```haskell
    {program_p_completion}
    ```

    Its argument type is  
    ```haskell
    t = {t}
    ```

    **Your task**: produce a new function `Q` that satisfies the system prompt requirements.  
    - Make sure `Q` has a different name (e.g. append a `'`).  
    - Avoid trivial symmetric rewrites—show a genuine alternative implementation.  
    - Do not include any extra commentary beyond the required `<think>…</think>` and the `**Generated Program `Q`:** block.
""").strip()

BOB_SYSTEM_PROMPT = textwrap.dedent("""
    You are a Haskell equivalence checker. Given P and Q:
      1. If ∀x. P x == Q x, output "Equivalent."
      2. Otherwise, output a concrete counterexample x and the two outputs.
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
""")



# New SFT Templates


# --- Code Execution ---

class CodeExecutor:
    """A wrapper for executing Haskell code and comparing results."""
    def __init__(self, sinq_instance, timeout: float = 60.0):
        self.sinq = sinq_instance
        self.timeout = timeout
        self.tmp_dir = tempfile.mkdtemp(prefix="haskell_executor_")

    def _create_haskell_program(self, program_code: str, main_body: str) -> str:
        """Builds a complete Haskell source file from a function definition and a main body."""
        prog = textwrap.dedent(program_code).strip()
        body = textwrap.dedent(main_body).rstrip()

        # No imports needed for the initial programs.
        imports = ""

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

        logger.info(f"Executing Program: \n\n{program}\n\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir) as f:
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
        status_p, out_p = self.execute(p, x)
        status_q, out_q = self.execute(q, x)

        logger.info(f"Status P: {status_p}")
        logger.info(f"Status Q: {status_q}")
        logger.info(f"Output P: {out_p}")
        logger.info(f"Output Q: {out_q}")

        # Case 1: Both succeeded but outputs are different
        if status_p == "success" and status_q == "success":
            return out_p != out_q

        # Case 2: One succeeded, the other didn't (implies divergence)
        if (status_p == "success" and status_q != "success") or \
           (status_q == "success" and status_p != "success"):
            return True

        # Case 3: Both failed, but in different ways (implies divergence)
        # This covers cases like one compile error, one runtime error, or one timeout, etc.
        if status_p != "success" and status_q != "success" and status_p != status_q:
            return True

        # If none of the above, they do not diverge (e.g., both compile_error, or both runtime_timeout)
        return False


    def check_compiles(self, program_code: str) -> bool:
        """Checks if a Haskell program compiles successfully."""
        main_body = 'main :: IO ()\nmain = putStrLn "compiles"'
        program = self._create_haskell_program(program_code, main_body)

        logger.info(f"Compiling Program: \n\n{program}\n\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hs', delete=True, dir=self.tmp_dir) as f:
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

    def generate_proof_body(self, program_p: str, program_q: str, func_name_p: str, func_name_q: str, arg_type: str, error_msg: str = None) -> str:
        """
        Ask Alice to fill in the proof. If error_msg is set, include it so she can fix mistakes.
        Returns just the body (the lines after '=').
        """
        program_p_content = textwrap.dedent(program_p)
        program_q_content = textwrap.dedent(program_q)
        # 1. Build a prompt that shows the reflected P and Q,
        #    and that asks for a sequence of ===, ?lemmas, and *** QED.
        proof_prompt = textwrap.dedent(f"""
        You are an expert Haskell/Liquid Haskell prover.
        Below are two reflected definitions:

        ```haskell
        {{-@ reflect {func_name_p} @-}}
        {program_p_content}

        {{-@ reflect {func_name_q} @-}}
        {program_q_content}
        ```

        Please write the body of

        ```haskell
        lemma_{func_name_p}_equiv :: x:{arg_type} -> {{ {func_name_p} x == {func_name_q} x }}
        lemma_{func_name_p}_equiv x
          =   /* PROOF BODY HERE */
        ```

        Use `===` for equational steps, `? someLemma …` for hints, and end with `*** QED`.

        {f"If Liquid Haskell failed with:\n```\n{error_msg}\n```\nplease fix your proof accordingly." if error_msg else ""}
        """).strip()

        # Call your LLM (Alice) with this as a single-turn prompt:
        response = self.sinq.call_alice_for_proof(proof_prompt)
        logger.info(f"Alice's raw proof generation response: {response}")

        # Check if Alice returned the full function definition
        match = re.search(r"lemma_.*?_equiv x\s*=\s*(.*)", response, re.DOTALL)
        if match:
            body = match.group(1).strip()
            logger.info(f"Extracted proof body from full definition:\n{body}")
            return body
        else:
            # Assume the entire response is the proof body
            logger.info("Assuming entire response is the proof body.")
            return response.strip()

    def verify_equivalence(self, program_p: str, program_q: str):
        """Verifies semantic equivalence using Liquid Haskell."""
        logger.info("Verifying equivalence with Liquid Haskell...")

        func_name_p = get_function_name(program_p)
        func_name_q = get_function_name(program_q)
        arg_type_p = get_function_arg_type(program_p)
        arg_type_q = get_function_arg_type(program_q)

        logger.info(f"Function name P: {func_name_p}")
        logger.info(f"Function name Q: {func_name_q}")
        logger.info(f"Argument type P: {arg_type_p}")
        logger.info(f"Argument type Q: {arg_type_q}")

        if not func_name_p or not arg_type_p or not func_name_q or not arg_type_q:
            logger.warning("Could not extract function name or argument type for P or Q. Skipping LH verification.")
            return "lh_error", "Function name or arg type extraction failed.", None

        # Create temporary files for Original.hs and Variant.hs
        original_file_path = os.path.join(self.tmp_dir, "Original.hs")
        dedented_program_p = textwrap.dedent(program_p).strip()
        with open(original_file_path, "w") as f_p:
            content_p = "module Original where\n" \
                        f"{{-@ reflect {func_name_p} @-}}\n" \
                        f"{dedented_program_p}"
            f_p.write(content_p)
        
        variant_file_path = os.path.join(self.tmp_dir, "Variant.hs")
        dedented_program_q = textwrap.dedent(program_q).strip()
        with open(variant_file_path, "w") as f_q:
            content_q = "module Variant where\n" \
                        f"{{-@ reflect {func_name_q} @-}}\n" \
                        f"{dedented_program_q}"
            f_q.write(content_q)

        # Create the Equiv.hs file
        equiv_code_template = textwrap.dedent("""
            {{-@ LIQUID "--reflection" @-}}
            {{-@ LIQUID "--ple" @-}}
            module Equiv where
            import Original ({func_name_p})
            import Variant  ({func_name_q})
            import Language.Haskell.Liquid.ProofCombinators

            {{-@ reflect {func_name_p} @-}}
            {program_p_content}

            {{-@ reflect {func_name_q} @-}}
            {program_q_content}

            -- Alice’s detailed proof of equivalence
            {{-@ lemma_{func_name_p}_equiv :: x:{arg_type_p} -> {{ {func_name_p} x == {func_name_q} x }} @-}}
            lemma_{func_name_p}_equiv :: {arg_type_p} -> Proof
            lemma_{func_name_p}_equiv x
              =   {proof_body}
        """)
        
        # 1. Fill the initial proof_body with a stub
        proof_body = f"{func_name_p} x === {func_name_q} x *** QED"
        MAX_PROOF_ATTEMPTS = 3 # Set a limit for proof attempts

        for attempt in range(MAX_PROOF_ATTEMPTS):
            logger.info(f"--- Proof attempt {attempt + 1}/{MAX_PROOF_ATTEMPTS} ---")
            
            # 2. Render Equiv.hs with proof_body
            equiv_code = equiv_code_template.format(
                func_name_p=func_name_p,
                func_name_q=func_name_q,
                program_p_content=dedented_program_p,
                program_q_content=dedented_program_q,
                arg_type_p=arg_type_p,
                proof_body=proof_body
            )

            equiv_file_path = os.path.join(self.tmp_dir, "Equiv.hs")
            with open(equiv_file_path, 'w') as f_equiv:
                f_equiv.write(equiv_code)
            
            try:
                # 3. Run Liquid Haskell
                env = os.environ.copy()
                env['PATH'] = f"{os.path.expanduser('~')}/.local/bin:{env['PATH']}"
                
                lh_process = subprocess.run(
                    [
                        'ghc',
                        '-fplugin=LiquidHaskell',
                        '-package',
                        'liquid-prelude',
                        equiv_file_path
                    ],
                    capture_output=True, text=True, timeout=self.timeout,
                    cwd=self.tmp_dir, 
                    env=env
                )
                
                logger.info(f"Equiv.hs (Attempt {attempt + 1}): \n{equiv_code}")
                
                if lh_process.returncode == 0:
                    logger.info("✅ Liquid Haskell proof accepted.")
                    return "proved", lh_process.stdout, None
                
                # 4. On failure, extract stderr and ask Alice for a new proof_body
                error_msg = lh_process.stderr
                logger.warning(f"❌ Liquid Haskell proof refuted (Attempt {attempt + 1}):\n{error_msg}")
                
                if attempt < MAX_PROOF_ATTEMPTS - 1:
                    logger.info("Asking Alice for a new proof body...")
                    proof_body = self.generate_proof_body(
                        program_p, program_q, func_name_p, func_name_q, arg_type_p, error_msg
                    )
                else:
                    logger.warning(f"Max proof attempts ({MAX_PROOF_ATTEMPTS}) reached. Final refutation.")
                    counterexample_match = re.search(r"Counterexample: (.*)", error_msg)
                    counterexample = counterexample_match.group(1).strip() if counterexample_match else None
                    return "refuted", error_msg, counterexample

            except subprocess.TimeoutExpired:
                logger.warning(f"Liquid Haskell verification timed out (Attempt {attempt + 1}).")
                # If it times out on the last attempt, return timeout
                if attempt == MAX_PROOF_ATTEMPTS - 1:
                    return "lh_timeout", "Liquid Haskell verification timed out", None
            except Exception as e:
                logger.error(f"An unexpected error occurred during Liquid Haskell verification: {e}")
                return "lh_error", str(e), None

        # This part should ideally not be reached if the loop handles all cases
        return "refuted", "Max proof attempts reached without success.", None


# --- SInQ Self-Play and Fine-tuning ---
class SavePeftModelCallback(TrainerCallback):
    def __init__(self, model_type=None, iteration=None):
        self.model_type = model_type
        self.iteration = iteration

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        output_dir = args.output_dir
        model = kwargs['model']
        
        if self.model_type is not None and self.iteration is not None:
            adapter_dir_name = f"{self.model_type}-adapter-iter-{self.iteration}-epoch-{int(epoch)}"
        else:
            adapter_dir_name = f"adapter-epoch-{int(epoch)}"

        adapter_path = os.path.join(output_dir, adapter_dir_name)
        model.save_pretrained(adapter_path)
        logger.info(f"Saved adapter for epoch {int(epoch)} to {adapter_path}")

class SInQ_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized_item = self.tokenizer(item, truncation=True, padding="max_length")
        return {k: torch.tensor(v) for k, v in tokenized_item.items()}

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

        self.base_model = None # Lazy load to save VRAM

        self.initial_adapter_path = args.initial_adapter_path
        if self.initial_adapter_path:
            dest_path = os.path.join(self.args.output_dir, "initial_adapter_base")
            if os.path.exists(self.initial_adapter_path):
                if not os.path.exists(dest_path):
                    logger.info(f"Copying initial adapter from {self.initial_adapter_path} to {dest_path}")
                    shutil.copytree(self.initial_adapter_path, dest_path)
            else:
                logger.warning(f"Initial adapter path specified but not found: {self.initial_adapter_path}")
                self.initial_adapter_path = None # Reset if not found

        self.executor = CodeExecutor(self, timeout=args.timeout)
        self.programs = self.load_initial_programs(args.dataset_name)
        self.alice_adapter_path = self.initial_adapter_path
        self.bob_adapter_path = self.initial_adapter_path
        self.cumulative_alice_training_data = []

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
        print_nvidia_smi("After initializing vLLM model")
        print_gpu_memory_usage("After initializing vLLM model")

    def load_initial_programs(self, dataset_name):
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
            elif dataset_name.endswith('.jsonl'):
                logger.info(f"Found local .jsonl file. Loading from: {dataset_name}")
                dataset = load_dataset('json', data_files={'train': dataset_name}, split='train', streaming=True)
                if self.args.num_initial_programs==0:
                    logger.info("`num_initial_programs` is not set, loading all programs from stream.")
                    dataset_iterator = dataset
                else:
                    logger.info(f"Loading {self.args.num_initial_programs} initial programs from stream.")
                    dataset_iterator = dataset.take(self.args.num_initial_programs)
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
                if 'code' in program_item:
                    programs.append({'code': program_item['code']})

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
            # print(f"Alice output: \n\n{text}\n\n")
            # Extract program Q
            program_q_match = re.search(r"\*\*Generated Program `Q`:\*\*\s*```haskell\n(.*?)\n```", text, re.DOTALL)
            if not program_q_match:
                logger.warning("🟥 --- Alice parsing failed: Could not find 'Generated Program Q' block ---")

                logger.warning(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            program_q = program_q_match.group(1).strip()
            
            if not program_q:
                logger.warning("🟥 --- Alice parsing failed: `Q` is empty after parsing ---")
                logger.warning(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            
            return program_q, text
        except Exception as e:
            logger.error(f"🟥 --- Alice parsing failed with exception: {e} ---")
            logger.error(f"Full output from Alice:\n{text}")
            return None, None

    def parse_bob_output(self, text):
        try:
            # If Bob claims equivalence
            if re.match(r"^Equivalent\.$", text.strip()):
                return "Equivalent", None, None
            
            # Extract counterexample if programs are inequivalent
            # More robust regex for LH counterexamples: typically 'Counterexample: <value> with input <input_value>'
            # or just 'Counterexample: <value>'
            counterexample_match = re.search(r"[Cc]ounterexample: (.*?)(?: with input (.*?))?", text)
            if counterexample_match:
                counterexample_value = counterexample_match.group(1).strip()
                # If there's a second group, it's the specific input for the counterexample.
                # Otherwise, the entire captured value is the counterexample.
                counterexample = counterexample_match.group(2).strip() if counterexample_match.group(2) else counterexample_value

                # Remove potential 'x = ' prefix from the diverging input.
                counterexample = re.sub(r'^[ \t]*\w+[ \t]*=[ \t]*', '', counterexample).strip()

                output_p_match = re.search(r"[oO]utput P:?\s*`(.*?)`", text)
                output_q_match = re.search(r"[oO]utput Q:?\s*`(.*?)`", text)

                output_p = output_p_match.group(1).strip() if output_p_match else None
                output_q = output_q_match.group(1).strip() if output_q_match else None

                return counterexample, output_p, output_q
            
            return None, None, None
        except Exception as e:
            logger.error(f"🟥 --- Bob parsing failed with exception: {e} ---")
            return None, None, None

    def run_alice(self, program_p_completion):
        """Alice generates a variant of a program."""
        logger.info(f"Running Alice...")
        
        # Extract the argument type for the original program P
        arg_type_p = get_function_arg_type(program_p_completion)
        if not arg_type_p:
            logger.warning("Could not determine argument type for program P. Using 'a' as a placeholder.")
            arg_type_p = "a" # Fallback to a generic type variable

        # ALICE_USER_PROMPT already contains {program}
        user_content = ALICE_USER_PROMPT.format(
            program_p_completion=program_p_completion,
            t=arg_type_p
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
        request_outputs = self.vllm_model.generate(
            [prompt] * 1,
            sampling_params,
            lora_request=LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
        )

        logger.info(f"Length of Alice request_outputs: {len(request_outputs)}")

        for output in request_outputs:
            result_text = output.outputs[0].text
            program_q, alice_raw_output = self.parse_alice_output(result_text)
            if program_q:
                return program_q, alice_raw_output, user_content
        
        return None, None, None

    def call_alice_for_proof(self, proof_prompt: str) -> str:
        """Generic method to call Alice with a specified prompt."""
        logger.info("Calling Alice for proof generation...")
        
        messages = [
            {"role": "system", "content": proof_prompt}
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
        
        request_outputs = self.vllm_model.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
        )

        if request_outputs:
            return request_outputs[0].outputs[0].text
        return ""

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
        valid_bob_inputs = []
        bob_training_responses = []

        # The single RequestOutput contains a list of n_samples CompletionOutputs
        bob_outputs = request_outputs[0].outputs
        for output in bob_outputs:
            result_text = output.text
            counterexample, output_p, output_q = self.parse_bob_output(result_text)

            logger.info(f"Bob's counterexample: {counterexample}")
            
            if counterexample and counterexample != "Equivalent":
                # For Bob, we only consider a 'correct' counterexample if it diverges on the concrete input
                if self.executor.check_divergence(program_p, program_q, counterexample):
                    n_correct += 1
                    valid_bob_inputs.append({
                        "counterexample": counterexample,
                        "output_p": output_p,
                        "output_q": output_q
                    })
                    # Add the raw successful response for Bob's training
                    bob_training_responses.append(result_text)
            elif counterexample == "Equivalent":
                # If Bob claims equivalence, and LH proved it, this is a correct Bob response
                # For now, we don't count this towards 'n_correct' for difficulty, but good for training.
                pass

        difficulty = 10 * (1 - (n_correct / self.args.n_samples))
        logger.info(f"Bob found {n_correct}/{self.args.n_samples} correct diverging inputs. Calculated difficulty: {difficulty:.2f}")
        
        return difficulty, valid_bob_inputs, bob_training_responses

    def _initialize_base_model_for_finetuning(self):
        if self.base_model is None:
            logger.info("Initializing base model for fine-tuning...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto"
            )

    def finetune_model(self, dataset, model_type, iteration):
        """Fine-tunes the model on the given dataset."""
        logger.info(f"Starting fine-tuning for {model_type}...")
        
        output_dir = os.path.join(self.args.output_dir, model_type)
        os.makedirs(output_dir, exist_ok=True)

        self._initialize_base_model_for_finetuning()

        # Enable gradient checkpointing on the base model BEFORE wrapping with PEFT
        self.base_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        current_adapter_to_load = None
        if model_type == "alice":
            current_adapter_to_load = self.alice_adapter_path
        elif model_type == "bob":
            current_adapter_to_load = self.bob_adapter_path

        if current_adapter_to_load and os.path.exists(current_adapter_to_load):
            logger.info(f"Loading adapter from {current_adapter_to_load} to continue fine-tuning.")
            peft_model = PeftModel.from_pretrained(self.base_model, current_adapter_to_load, is_trainable=True)
        else:
            if current_adapter_to_load:
                logger.warning(f"Adapter path {current_adapter_to_load} not found. Creating a new adapter.")
            logger.info("Creating a new PeftModel for training from base model.")
            peft_model = get_peft_model(self.base_model, lora_config)

        peft_model.enable_input_require_grads()

        # Recommended for gradient checkpointing
        peft_model.config.use_cache = False

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            learning_rate=self.args.learning_rate,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=0, # Defer saving to callback
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=SInQ_Dataset(dataset, self.tokenizer),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[SavePeftModelCallback(model_type=model_type, iteration=iteration)]
        )

        # Print GPU memory usage before training
        print_nvidia_smi("Before training")
        print_gpu_memory_usage(f"Before training '{model_type}' model")

        trainer.train()
        
        print_nvidia_smi("After training")
        print_gpu_memory_usage(f"After training '{model_type}' model")

        adapter_dir_name = f"{model_type}-adapter-iter-{iteration}-epoch-{int(self.args.num_train_epochs)}"
        adapter_path = os.path.join(output_dir, adapter_dir_name)
        if model_type == "alice":
            self.alice_adapter_path = adapter_path
        else:
            self.bob_adapter_path = adapter_path
        
        torch.cuda.empty_cache()
        # print_gpu_memory_usage(f"After cleaning up fine-tuning model")

    def run_self_play_loop(self):
        logger.info("Starting self-play loop...")
        
        os.makedirs(os.path.join(self.args.output_dir, "seq_data"), exist_ok=True)

        for i in range(self.args.n_iterations):
            logger.info(f"--- Self-Play Iteration {i+1}/{self.args.n_iterations} ---")
            
            run_bob_triggered_count = 0
            
            warning_counts = {
                "p_compile_fail": 0,
                "alice_generation_fail": 0,
                "q_compile_fail": 0,
                "alice_not_divergent": 0,
                "lh_error": 0,
                "lh_timeout": 0,
            }
            
            iteration_data = [] # To store JSON records for this iteration
            bob_training_data = []
            
            # random.shuffle(self.programs)
            
            for j, program_item in enumerate(tqdm(self.programs, desc=f"Iteration {i+1}")):

                p_original_code = program_item['code']
                p_original_code = p_original_code.replace("main :: IO ()", "")
                p_original_code = strip_comments(p_original_code)


                tqdm.write(f"\n--- Processing example {j+1}/{len(self.programs)} in Iteration {i+1} ---")
                # tqdm.write(f"Original program (P):\n{p_original_code}")

                if not self.executor.check_compiles(p_original_code):
                    logger.warning("🟨 Original program P failed to compile. Skipping example.")
                    warning_counts["p_compile_fail"] += 1
                    continue

                # Alice's turn
                q_candidate, alice_raw_output, alice_user_content = self.run_alice(p_original_code)
                
                if not q_candidate:
                    logger.warning("🟨 Alice failed to generate a candidate program.")
                    warning_counts["alice_generation_fail"] += 1
                    continue

                # Check if Q is different from P
                if q_candidate.strip() == p_original_code.strip():
                    logger.warning("🟨 Alice generated an identical program Q to P. Skipping example.")
                    warning_counts["alice_not_divergent"] += 1
                    continue

                if not self.executor.check_compiles(q_candidate):
                    logger.warning("🟨 Candidate Q failed to compile. Skipping example.")
                    warning_counts["q_compile_fail"] += 1
                    continue
            
                logger.info(f"Alice's candidate program: \n{q_candidate}")

                # Phase 2: Embedding & Proving with Liquid Haskell
                lh_status, lh_output, lh_counterexample = self.executor.verify_equivalence(p_original_code, q_candidate)
                
                record = {
                    "p": p_original_code,
                    "q": q_candidate,
                    "status": lh_status,
                    "lh_output": lh_output
                }

                if lh_status == "proved":
                    logger.info("✅ Alice generated an equivalent program with a successful LH proof.")
                    iteration_data.append(record)

                    # Add to Alice's training data (proof-conditioned)
                    messages = [
                        {"role": "system", "content": ALICE_SYSTEM_PROMPT},
                        {"role": "user", "content": alice_user_content},
                        {"role": "assistant", "content": alice_raw_output}
                    ]
                    self.cumulative_alice_training_data.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

                elif lh_status == "refuted":
                    logger.warning("❌ Alice's claim was refuted by Liquid Haskell.")
                    record["counterexample"] = lh_counterexample
                    iteration_data.append(record)
                    warning_counts["alice_not_divergent"] += 1 # Renaming this for LH refutations

                    # Phase 3: Bob vs. Liquid Haskell (optional sanity check & training)
                    run_bob_triggered_count += 1
                    # The difficulty is now how easily Bob finds a counterexample for Alice's *refuted* claim.
                    difficulty, valid_bob_inputs, bob_successful_responses = self.run_bob(p_original_code, q_candidate)
                    
                    # Store Bob's successful attempts for evaluation and training
                    record["bob_difficulty"] = difficulty
                    record["valid_bob_inputs"] = valid_bob_inputs
                    record["bob_successful_responses"] = bob_successful_responses # Store for analysis if needed

                    # In SEQ, Alice should be trained to *avoid* refutations, or to produce *provable* equivalents.
                    # For now, we are focusing Alice's training on proved examples. If she is refuted, it's a negative example.
                    # We could add a separate mechanism for Alice to learn from refutations.

                    # Create Bob's training data from his successful attempts or LH's counterexample
                    for bob_response in bob_successful_responses:
                        # Bob's training data is based on successfully finding a counterexample
                        # The original `BOB_SYSTEM_PROMPT` is used for Bob's generation, so we use that for his training context.
                        messages = [
                            {"role": "system", "content": BOB_SYSTEM_PROMPT},
                            {"role": "user", "content": BOB_USER_PROMPT_TEMPLATE.format(program_p=p_original_code, program_q=q_candidate)},
                            {"role": "assistant", "content": bob_response}
                        ]
                        bob_training_data.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

                    if not valid_bob_inputs and lh_counterexample:
                        logger.info("🟨 Bob failed to find any counterexample, but LH provided one. Creating a hard training example for Bob from LH's counterexample.")
                        # Use BOB_COUNTEREXAMPLE_SFT_PROMPT for this hard training example
                        # The original BOB_COUNTEREXAMPLE_SFT_PROMPT was removed, use BOB_SYSTEM_PROMPT instead for system content
                        bob_completion = f"Counterexample: `{lh_counterexample}`\nOutput P: `...`\nOutput Q: `...`" # Placeholder for outputs
                        messages = [
                            {"role": "system", "content": BOB_SYSTEM_PROMPT},
                            {"role": "user", "content": BOB_USER_PROMPT_TEMPLATE.format(program_p=p_original_code, program_q=q_candidate)},
                            {"role": "assistant", "content": bob_completion}
                        ]
                        bob_training_data.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

                elif lh_status == "lh_timeout":
                    logger.warning("⏱️ Liquid Haskell verification timed out. Skipping example.")
                    warning_counts["lh_timeout"] += 1
                    # Do NOT append to iteration_data or training data, as it's an unreliable example.
                    continue
                else: # lh_error
                    logger.error("🔴 Liquid Haskell verification failed unexpectedly. Skipping example.")
                    warning_counts["lh_error"] += 1
                    # Do NOT append to iteration_data or training data.
                    continue

            # Save iteration data to JSON
            output_file_path = os.path.join(self.args.output_dir, "seq_data", f"iteration_{i}.json")
            with open(output_file_path, 'w') as f:
                json.dump(iteration_data, f, indent=4)
            logger.info(f"Saved iteration {i+1} data to {output_file_path}")

            # --- Phase 4: Data Aggregation & Fine-Tuning ---
            # Alice's training data comes from successfully proved LH claims.
            # Bob's training data comes from his successful counterexamples, and LH-refuted cases.

            logger.info(f"Iteration {i+1} summary:")
            logger.info(f"  - Bob was triggered {run_bob_triggered_count} times.")
            logger.info(f"  - Total processed examples: {len(iteration_data)}")
            logger.info(f"  - LH Proved examples: {len([r for r in iteration_data if r['status'] == 'proved'])}")
            logger.info(f"  - LH Refuted examples: {len([r for r in iteration_data if r['status'] == 'refuted'])}")
            logger.info(f"  - LH Timeout examples: {len([r for r in iteration_data if r['status'] == 'lh_timeout'])}")
            logger.info(f"  - LH Error examples: {len([r for r in iteration_data if r['status'] == 'lh_error'])}")
            logger.info(f"  - Cumulative Alice training data size: {len(self.cumulative_alice_training_data)}")
            logger.info(f"  - Bob training data size: {len(bob_training_data)}")
            logger.info("  - 🟥 🟥 🟥 Warning counts: 🟥 🟥 🟥")
            logger.info(f"    - Original program P failed to compile: {warning_counts['p_compile_fail']}")
            logger.info(f"    - Alice failed to generate a candidate program/refinement stub: {warning_counts['alice_generation_fail']}")
            logger.info(f"    - Candidate Q failed to compile: {warning_counts['q_compile_fail']}")
            logger.info(f"    - Alice's claim refuted by Liquid Haskell: {warning_counts['alice_not_divergent']}")
            logger.info(f"    - Liquid Haskell verification timed out: {warning_counts['lh_timeout']}")
            logger.info(f"    - Liquid Haskell verification failed unexpectedly: {warning_counts['lh_error']}")

            if self.cumulative_alice_training_data or bob_training_data:
                logger.info("Releasing vLLM model to free up memory for fine-tuning...")
                del self.vllm_model
                torch.cuda.empty_cache()

                print_nvidia_smi("After releasing vLLM model")
                print_gpu_memory_usage("After releasing vLLM model")


                if self.cumulative_alice_training_data:
                    self.finetune_model(self.cumulative_alice_training_data, "alice", i)

                if bob_training_data:
                    self.finetune_model(bob_training_data, "bob", i)
                
                # Release base model memory
                del self.base_model
                self.base_model = None
                torch.cuda.empty_cache()

                logger.info("Re-initializing vLLM model for the next generation round...")
                self._initialize_vllm()
            
            # New programs are not added based on difficulty anymore; they are just processed.
            # self.programs.extend(new_programs) # This was for adding 'hard' programs for Alice to learn from.
            # In SEQ, Alice learns from successfully proved examples.

            if self.args.run_evaluation:
                self.evaluate(i)

    def evaluate(self, iteration):
        logger.info(f"Starting final evaluation for iteration {iteration}...")
        
        # Evaluate Alice's model
        if self.alice_adapter_path:
            logger.info(f"Evaluating Alice's model with adapter path: {self.alice_adapter_path}")
            self.evaluate_agent("alice", self.alice_adapter_path, iteration)
        else:
            logger.info("Alice adapter not found, skipping evaluation.")

    def evaluate_agent(self, agent_name, adapter_path, iteration):
        logger.info(f"--- Evaluating {agent_name}'s Model for iteration {iteration} ---")
        
        if not (adapter_path and os.path.exists(adapter_path)):
            logger.warning(f"Adapter path for {agent_name} does not exist: {adapter_path}. Skipping evaluation.")
            return

        logger.info(f"Adapter path for evaluation: {adapter_path}")
        
        # === HumanEval Evaluation ===
        eval_working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Evaluation/HumanEval'))
        eval_script_path = os.path.join(eval_working_dir, 'eval_script/eval_adapter.sh')

        adapter_path_abs = os.path.abspath(adapter_path)
        
        logger.info(f"Adapter absolute path: {adapter_path_abs}")
        logger.info(f"Eval script path: {eval_script_path}")
        logger.info(f"Working directory: {eval_working_dir}")

        logger.info(f"Submitting evaluation script for {agent_name} via sbatch...")

        subprocess.run(['sbatch', eval_script_path, adapter_path_abs, "hs", self.model_name, str(self.args.n_humaneval_evaluations_per_iteration)], check=True, cwd=eval_working_dir)
        logger.info(f"Submitted Haskell evaluation script for {agent_name} via sbatch.")
        subprocess.run(['sbatch', eval_script_path, adapter_path_abs, "python", self.model_name, str(self.args.n_humaneval_evaluations_per_iteration)], check=True, cwd=eval_working_dir)
        logger.info(f"Submitted Python evaluation script for {agent_name} via sbatch.")


        # === LiveCodeBench Evaluation ===
        eval_working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Evaluation/LiveCodeBench/code_generation_lite'))
        eval_script_path = os.path.join(eval_working_dir, 'eval_livecodebench_adapter.sh')

        subprocess.run(['sbatch', eval_script_path, adapter_path_abs, self.model_name], check=True, cwd=eval_working_dir)
        logger.info(f"LiveCodeBench Evaluation script for {agent_name} submitted.")

        # === New SEQ Evaluation Metrics ===
        logger.info(f"--- Calculating SEQ Evaluation Metrics for iteration {iteration} ---")
        
        iteration_data_path = os.path.join(self.args.output_dir, "seq_data", f"iteration_{iteration}.json")
        if not os.path.exists(iteration_data_path):
            logger.warning(f"Iteration data file not found: {iteration_data_path}. Skipping SEQ evaluation metrics.")
            return

        with open(iteration_data_path, 'r') as f:
            iteration_data = json.load(f)
        
        total_examples = len(iteration_data)
        if total_examples == 0:
            logger.info("No examples processed in this iteration. Skipping SEQ evaluation metrics.")
            return

        # Proof Accuracy
        proved_count = len([r for r in iteration_data if r['status'] == 'proved'])
        proof_accuracy = (proved_count / total_examples) * 100 if total_examples > 0 else 0
        logger.info(f"Proof Accuracy (LH Proved): {proof_accuracy:.2f}%")

        # Counterexample Accuracy (Bob vs. LH)
        refuted_examples = [r for r in iteration_data if r['status'] == 'refuted']
        total_refuted = len(refuted_examples)
        
        bob_matches_lh_counterexample = 0
        if total_refuted > 0:
            for record in refuted_examples:
                lh_ce = record.get("counterexample")
                bob_valid_inputs = record.get("valid_bob_inputs", [])

                if lh_ce and bob_valid_inputs:
                    # Check if any of Bob's valid inputs match LH's counterexample (simple string match for now)
                    # In a real scenario, this would involve canonicalization or re-execution to compare semantic equivalence of counterexamples
                    for bob_input in bob_valid_inputs:
                        if bob_input.get("counterexample") == lh_ce:
                            bob_matches_lh_counterexample += 1
                            break # Only need one match per refuted example
            
            counterexample_accuracy = (bob_matches_lh_counterexample / total_refuted) * 100
            logger.info(f"Counterexample Accuracy (Bob matches LH): {counterexample_accuracy:.2f}%")
        else:
            logger.info("No refuted examples found. Skipping Counterexample Accuracy calculation.")


def main():
    parser = argparse.ArgumentParser(description="SInQ Self-Play and Fine-tuning")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")

    parser.add_argument('--dataset_name', type=str, default='../data/successfully_compiled_sorted_haskell_dataset', help="Hugging Face dataset name or local path for initial programs.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save models and results.")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens for the model's context window and generation.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for sampling.")
    parser.add_argument('--top_k', type=int, default=20, help="Top-k for sampling.")
    parser.add_argument('--presence_penalty', type=float, default=1.5, help="Presence penalty for sampling.")
    parser.add_argument('--timeout', type=float, default=20.0, help="Timeout for code execution.")
    parser.add_argument('--n_iterations', type=int, default=10, help="Number of self-play iterations.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples for Alice to generate.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for fine-tuning.")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r.")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--num_initial_programs', type=int, default=100, help="Number of initial programs to load.")
    parser.add_argument('--difficulty_threshold', type=float, default=0.0, help="Difficulty threshold for filtering Alice's training data.")
    parser.add_argument('--n_humaneval_evaluations_per_iteration', type=int, default=1, help="Number of evaluations to run per iteration.")
    parser.add_argument('--run_evaluation', action='store_true', help="Whether to run evaluation after each iteration.")
    parser.add_argument('--initial_adapter_path', type=str, default=None, help="Path to an initial LoRA adapter to continue fine-tuning from.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of tensor parallel processes.")
    
    args = parser.parse_args()

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    sinq = SInQ(args)

    try:
        sinq.run_self_play_loop()
    finally:
        # Explicitly clean up the temporary directory created by CodeExecutor
        if os.path.exists(sinq.executor.tmp_dir):
            logger.info(f"Cleaning up temporary directory: {sinq.executor.tmp_dir}")
            shutil.rmtree(sinq.executor.tmp_dir)

if __name__ == "__main__":
    main()



