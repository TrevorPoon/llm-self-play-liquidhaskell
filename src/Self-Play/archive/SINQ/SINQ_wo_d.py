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
    You are an expert Haskell programmer. Your task is to generate a semantically inequivalent variant of a given Haskell program,.
    You must also provide a diverging input, which is a valid input for both programs, but on which they produce different outputs.
                                      
    A good inequivalent program `Q` should be subtly different from `P`.
    A good diverging input `x` should be simple and clearly demonstrate the semantic difference between `P` and `Q`.

    First, think step-by-step and write down your analysis of program `P` and your strategy for creating an inequivalent program `Q`. Enclose this reasoning within `<think>` and `</think>` tags.
    After the thinking block, the final answer could **only** be in the following format, without any additional explanation or context.

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
    Original program `P`:
    ```haskell
    {program}
    ```
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
""")

ALICE_DIFFICULTY_PREDICTION_PROMPT_TEMPLATE = textwrap.dedent("""
    Difficulty level: Any
    ```haskell
    {program}
    ```
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
        
        if status_p == status_q == "runtime_error":
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

        self.executor = CodeExecutor(timeout=args.timeout)
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
        logger.info("After initializing vLLM model (nvidia-smi):")
        print_nvidia_smi(f"After initializing vLLM model")
        logger.info("After initializing vLLM model (GPU memory usage):")
        print_gpu_memory_usage(f"After initializing vLLM model")

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
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            save_strategy="epoch",
            save_steps=500,
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=SInQ_Dataset(dataset, self.tokenizer),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        # Print GPU memory usage before training
        logger.info("Before training (nvidia-smi):")
        print_nvidia_smi("Before training")
        logger.info(f"Before training '{model_type}' model (GPU memory usage):")
        print_gpu_memory_usage(f"Before training '{model_type}' model")

        trainer.train()
        
        logger.info("After training (nvidia-smi):")
        print_nvidia_smi("After training")
        logger.info(f"After training '{model_type}' model (GPU memory usage):")
        print_gpu_memory_usage(f"After training '{model_type}' model")

        adapter_dir_name = f"{model_type}-adapter-iter-{iteration}-epoch-{int(self.args.num_train_epochs)}"
        adapter_path = os.path.join(output_dir, adapter_dir_name)
        if model_type == "alice":
            self.alice_adapter_path = adapter_path
        else:
            self.bob_adapter_path = adapter_path
        
        torch.cuda.empty_cache()
        # logger.info(f"After cleaning up fine-tuning model")

    def run_self_play_loop(self):
        logger.info("Starting self-play loop...")
        
        for i in range(self.args.n_iterations):
            logger.info(f"--- Self-Play Iteration {i+1}/{self.args.n_iterations} ---")
            
            run_bob_triggered_count = 0
            
            warning_counts = {
                "p_compile_fail": 0,
                "alice_generation_fail": 0,
                "q_compile_fail": 0,
                "alice_not_divergent": 0,
            }
            
            candidate_examples = []
            bob_training_data = []
            
            # random.shuffle(self.programs)
            
            for j, p_original in enumerate(tqdm(self.programs, desc=f"Iteration {i+1}")):
                tqdm.write(f"\n--- Processing example {j+1}/{len(self.programs)} in Iteration {i+1} ---")

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
                    messages = [
                        {"role": "system", "content": BOB_SYSTEM_PROMPT},
                        {"role": "user", "content": BOB_USER_PROMPT_TEMPLATE.format(program_p=p_original, program_q=q_candidate)},
                        {"role": "assistant", "content": bob_response}
                    ]
                    bob_training_data.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

                # If Bob completely failed, but Alice provided a valid input, create a hard training example for Bob
                if not valid_bob_inputs:
                    logger.info("🟨 Bob failed to find any diverging input. Creating a hard training example for Bob from Alice's input.")
                    bob_completion = f"\n**Analysis:**\n<analysis>\n\n**Diverging Input `x`:**\n```\n{x_candidate}\n```"
                    messages = [
                        {"role": "system", "content": BOB_SYSTEM_PROMPT},
                        {"role": "user", "content": BOB_USER_PROMPT_TEMPLATE.format(program_p=p_original, program_q=q_candidate)},
                        {"role": "assistant", "content": bob_completion}
                    ]
                    bob_training_data.append(self.tokenizer.apply_chat_template(messages, tokenize=False))
            
            # --- Task 3: Biasing Alice's Training Set ---
            hard_examples = [ex for ex in candidate_examples if ex["difficulty"] >= self.args.difficulty_threshold]
            easy_examples = [ex for ex in candidate_examples if ex["difficulty"] < self.args.difficulty_threshold]
            
            new_programs = [ex["q_candidate"] for ex in hard_examples]

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
            
            new_alice_training_data = []
            for ex in final_alice_examples:
                # --- (A) Main SFT Example ---
                main_sft_user_content = ALICE_USER_PROMPT.format(
                    program=ex['p_original']
                )
                main_sft_messages = [
                    {"role": "system", "content": ALICE_SYSTEM_PROMPT},
                    {"role": "user", "content": main_sft_user_content},
                    {"role": "assistant", "content": ex['alice_raw_output']}
                ]
                new_alice_training_data.append(self.tokenizer.apply_chat_template(main_sft_messages, tokenize=False))
            
            self.cumulative_alice_training_data.extend(new_alice_training_data)
            
            logger.info(f"Iteration {i+1} summary:")
            logger.info(f"  - Bob was triggered {run_bob_triggered_count} times.")
            logger.info(f"  - Total candidates generated: {len(candidate_examples)}")
            logger.info(f"  - Hard examples (d >= {self.args.difficulty_threshold}): {len(hard_examples)}")
            logger.info(f"  - Easy examples (d < {self.args.difficulty_threshold}): {len(easy_examples)}")
            logger.info(f"  - Sampled easy examples: {len(sampled_easies)}")
            logger.info(f"  - Alice training data size this iteration: {len(new_alice_training_data)}")
            logger.info(f"  - Average difficulty of hard examples: {sum([ex['difficulty'] for ex in hard_examples]) / len(hard_examples) if hard_examples else 0:.2f}")   
            logger.info(f"  - Average difficulty of easy examples: {sum([ex['difficulty'] for ex in easy_examples]) / len(easy_examples) if easy_examples else 0:.2f}")
            logger.info(f"  - Average difficulty of all examples: {sum([ex['difficulty'] for ex in candidate_examples]) / len(candidate_examples) if candidate_examples else 0:.2f}")
            logger.info(f"  - Cumulative Alice training data size: {len(self.cumulative_alice_training_data)}")
            logger.info(f"  - Bob training data size: {len(bob_training_data)}")
            logger.info("  - 🟥 🟥 🟥 Warning counts: 🟥 🟥 🟥")
            logger.info(f"    - Original program P failed to compile: {warning_counts['p_compile_fail']}")
            logger.info(f"    - Alice failed to generate a candidate program: {warning_counts['alice_generation_fail']}")
            logger.info(f"    - Candidate Q failed to compile: {warning_counts['q_compile_fail']}")
            logger.info(f"    - Alice's candidate not divergent / Execution failed: {warning_counts['alice_not_divergent']}")

            if self.cumulative_alice_training_data or bob_training_data:
                logger.info("Releasing vLLM model to free up memory for fine-tuning...")
                del self.vllm_model
                torch.cuda.empty_cache()

                logger.info("After releasing vLLM model (nvidia-smi):")
                print_nvidia_smi("After releasing vLLM model")
                logger.info("After releasing vLLM model (GPU memory usage):")
                print_gpu_memory_usage("After releasing vLLM model")


                if self.cumulative_alice_training_data:
                    self.finetune_model(self.cumulative_alice_training_data, "alice", i)

                if bob_training_data:
                    # self.finetune_model(bob_training_data, "bob", i)
                    pass

                logg
                
                # Release base model memory
                del self.base_model
                self.base_model = None
                torch.cuda.empty_cache()

                logger.info("Re-initializing vLLM model for the next generation round...")
                self._initialize_vllm()
            
            self.programs.extend(new_programs)

            if self.args.run_evaluation:
                logger.info(f"Starting final evaluation for iteration {i}...")
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
    parser.add_argument('--num_initial_programs', type=int, default=0, help="Number of initial programs to load.")
    parser.add_argument('--difficulty_threshold', type=float, default=0.0, help="Difficulty threshold for filtering Alice's training data.")
    parser.add_argument('--n_humaneval_evaluations_per_iteration', type=int, default=1, help="Number of evaluations to run per iteration.")
    parser.add_argument('--run_evaluation', action='store_true', help="Whether to run evaluation after each iteration.")
    parser.add_argument('--initial_adapter_path', type=str, default=None, help="Path to an initial LoRA adapter to continue fine-tuning from.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of tensor parallel processes.")
    parser.add_argument('--max_alice_retries', type=int, default=5, help="Maximum number of retries for Alice to generate a compilable program.")
    
    args = parser.parse_args()

    logger.info("Arguments:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    sinq = SInQ(args)
    sinq.run_self_play_loop()

if __name__ == "__main__":
    main()



