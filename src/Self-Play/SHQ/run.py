# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import random
import logging
import argparse
import subprocess
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import vllm
from torch.utils.data import Dataset
import re

from utils.execution import time_limit, create_tempdir
from utils.utils import extract_generation_code, get_function_name, print_nvidia_smi, print_gpu_memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Prompts (from SInQ Paper, Appendix C) ---

ALICE_PROMPT_TEMPLATE = """
You are an expert Haskell programmer. Your task is to generate a semantically inequivalent variant of a given Haskell program.
You must also provide a diverging input, which is a valid input for both programs, but on which they produce different outputs.

Original program `P`:
```haskell
{program}
```

A good inequivalent program `Q` should be subtly different from `P`.
A good diverging input `x` should be simple and clearly demonstrate the semantic difference between `P` and `Q`.

Respond with your thought process, the generated program `Q`, and the diverging input `x`.
The response should be in the following markdown format:

**Analysis:**
<Your analysis of the program `P` and the strategy to create an inequivalent program `Q`.>

**Generated Program `Q`:**
```haskell
<Your generated Haskell code for `Q`>
```

**Diverging Input `x`:**
```
<The diverging input `x`>
```
"""

BOB_PROMPT_TEMPLATE = """
You are an expert Haskell programmer. You are given two Haskell programs, `P` and `Q`.
Your task is to determine if they are semantically equivalent.
If they are inequivalent, you must provide a diverging input `x` on which `P(x) != Q(x)`.

Program `P`:
```haskell
{program_p}
```

Program `Q`:
```haskell
{program_q}
```

If the programs are equivalent, respond with "The programs are equivalent."
If they are inequivalent, respond with your thought process and the diverging input in the following markdown format:

**Analysis:**
<Your analysis of the differences between `P` and `Q`.>

**Diverging Input `x`:**
```
<The diverging input `x`>
```
"""

DIFFICULTY_PREDICTION_PROMPT_TEMPLATE = "difficulty: Any -> difficulty: {difficulty}"


# --- Code Execution ---

class CodeExecutor:
    """A wrapper for executing Haskell code and comparing results."""
    def __init__(self, timeout=10.0):
        self.timeout = timeout

    def execute(self, program_code, input_str):
        """Executes a Haskell program with a given input."""
        
        function_name = get_function_name(program_code)
        if not function_name:
            return "error", "Could not find function name in program."

        haskell_program = f"""
{program_code}

main :: IO ()
main = do
    let result = {function_name} ({input_str})
    print result
"""
        
        with create_tempdir() as tmp_dir:
            file_path = os.path.join(tmp_dir, "Main.hs")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(haskell_program)
            
            try:
                with time_limit(self.timeout):
                    compile_process = subprocess.run(
                        ['ghc', '-o', os.path.join(tmp_dir, 'main'), file_path],
                        capture_output=True
                    )

                    if compile_process.returncode != 0:
                        err_msg = f"Compilation failed: {compile_process.stderr.decode('utf-8', 'ignore')}"
                        print(err_msg)
                        return "error", err_msg

                    execute_process = subprocess.run(
                        [os.path.join(tmp_dir, 'main')],
                        capture_output=True,
                    )

                    if execute_process.returncode == 0:
                        output = execute_process.stdout.decode('utf-8', 'ignore').strip()
                        print(f"Execution success, output: {output}")
                        return "success", output
                    else:
                        err_msg = f"Execution failed: {execute_process.stderr.decode('utf-8', 'ignore')}"
                        print(err_msg)
                        return "error", err_msg
            except Exception as e:
                print(f"Execution resulted in exception: {e}")
                return "error", str(e)

    def check_divergence(self, p, q, x):
        """Checks if P(x) != Q(x)."""
        print(f"\n--- Checking divergence for input: {x} ---")
        print("--- Program P ---")
        print(p)
        status_p, out_p = self.execute(p, x)
        print("--- Program Q ---")
        print(q)
        status_q, out_q = self.execute(q, x)
        
        if status_p == "success" and status_q == "success":
            return out_p != out_q
        
        if status_p != status_q:
            return True
            
        if status_p != "success" and out_p != out_q:
             return True

        return False


# --- SInQ Self-Play and Fine-tuning ---

class SavePeftModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        output_dir = args.output_dir
        model = kwargs['model']
        
        adapter_weights_path = os.path.join(output_dir, f"adapter-weights-epoch-{int(epoch)}.pt")
        trainable_weights = {k: v for k, v in model.state_dict().items() if v.requires_grad}
        torch.save(trainable_weights, adapter_weights_path)
        print(f"Saved adapter weights for epoch {int(epoch)} to {adapter_weights_path}")

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
        
        self._initialize_vllm()

        self.base_model = None # Will be loaded on demand

        self.executor = CodeExecutor(timeout=args.timeout)
        self.programs = self.load_initial_programs(args.train_filename)
        self.alice_adapter_path = None
        self.bob_adapter_path = None

    def _initialize_vllm(self):
        logger.info("Initializing vLLM model...")
        self.vllm_model = vllm.LLM(
            model=self.model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            enable_lora=True,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_model_len,
        )
        print_gpu_memory_usage("After initializing vLLM model")

        print_nvidia_smi("After initializing vLLM model")

    def load_initial_programs(self, file_path):
        logger.info(f"Loading initial programs from {file_path}...")
        programs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                program = line.strip()
                if program.startswith('<s>'):
                    program = program[3:].lstrip()
                if program.endswith('</s>'):
                    program = program[:-4].rstrip()
                program = program.replace('<EOL>', '\n')
                programs.append(program)
        print(f"Loaded {len(programs)} programs.")
        return programs[:10]

    def parse_alice_output(self, text):
        try:
            # Extract program Q
            program_q_match = re.search(r"\*+Generated Program `?Q`?:\*+\s*```haskell\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
            if not program_q_match:
                print("🟥 --- Alice parsing failed: Could not find 'Generated Program Q' block ---")
                print(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            program_q = program_q_match.group(1).strip()

            # Extract diverging input x
            diverging_input_match = re.search(r"\*+Diverging Input `?x`?:\*+\s*```(?:haskell)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
            if not diverging_input_match:
                print("🟥 --- Alice parsing failed: Could not find 'Diverging Input x' block ---")
                print(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            diverging_input = diverging_input_match.group(1).strip()
            
            if not program_q or not diverging_input:
                print("🟥 --- Alice parsing failed: `Q` or `x` is empty after parsing ---")
                print(f"To solve this, please check the full output from Alice below and see why it failed to parse:\n{text}")
                return None, None
            
            return program_q, diverging_input
        except Exception as e:
            print(f"🟥 --- Alice parsing failed with exception: {e} ---")
            print(f"Full output from Alice:\n{text}")
            return None, None

    def parse_bob_output(self, text):
        try:
            # Extract diverging input
            diverging_input_match = re.search(r"\*+Diverging Input `?x`?:\*+\s*```(?:haskell)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
            if diverging_input_match:
                return diverging_input_match.group(1).strip()
            return None
        except Exception as e:
            print(f"🟥 --- Bob parsing failed with exception: {e} ---")
            return None

    def run_alice(self, program_p):
        """Alice generates a variant of a program."""
        logger.info("Running Alice...")
        
        prompt = ALICE_PROMPT_TEMPLATE.format(program=program_p)
        
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
            [prompt] * self.args.n_samples,
            sampling_params,
            lora_request=vllm.LoRARequest("alice_adapter", 1, self.alice_adapter_path) if self.alice_adapter_path else None
        )

        for output in request_outputs:
            result_text = output.outputs[0].text
            program_q, diverging_input = self.parse_alice_output(result_text)
            if program_q and diverging_input:
                return program_q, diverging_input, result_text
        
        return None, None, None

    def run_bob(self, program_p, program_q):
        """Bob checks for semantic equivalence."""
        logger.info("Running Bob...")
        prompt = BOB_PROMPT_TEMPLATE.format(program_p=program_p, program_q=program_q)
        
        sampling_params = vllm.SamplingParams(
            n=self.args.n_samples,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_tokens,
            presence_penalty=self.args.presence_penalty
        )
        
        request_outputs = self.vllm_model.generate(
            [prompt] * self.args.n_samples,
            sampling_params,
            lora_request=vllm.LoRARequest("bob_adapter", 1, self.bob_adapter_path) if self.bob_adapter_path else None
        )
        
        result_text = request_outputs[0].outputs[0].text
        diverging_input = self.parse_bob_output(result_text)
        return diverging_input, result_text

    def calculate_difficulty(self, p, q, n_attempts=10):
        # This is a placeholder. A real implementation would involve more sophisticated metrics.
        return "easy"

    def finetune_model(self, dataset, model_type):
        """Fine-tunes the model on the given dataset."""
        logger.info(f"Starting fine-tuning for {model_type}...")
        
        output_dir = os.path.join(self.args.output_dir, model_type)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Loading base model for fine-tuning...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        print_gpu_memory_usage("After loading base model for training")

        # Enable gradient checkpointing on the base model BEFORE wrapping with PEFT
        self.base_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
        )
        
        peft_model = get_peft_model(self.base_model, lora_config)

        # Load existing adapter weights if available for continued training
        adapter_path_to_load = None
        if model_type == "alice" and self.alice_adapter_path and os.path.exists(self.alice_adapter_path):
            adapter_path_to_load = self.alice_adapter_path
        elif model_type == "bob" and self.bob_adapter_path and os.path.exists(self.bob_adapter_path):
            adapter_path_to_load = self.bob_adapter_path

        if adapter_path_to_load:
            logger.info(f"Loading existing adapter weights from {adapter_path_to_load} for continued training.")
            # Ensure the model and weights are on the same device before loading
            adapter_weights = torch.load(adapter_path_to_load, map_location='cpu')
            peft_model.load_state_dict(adapter_weights, strict=False)
            logger.info("Successfully loaded adapter weights for continued training.")

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
            callbacks=[SavePeftModelCallback()]
        )

        # Print GPU memory usage before training
        print_nvidia_smi("Before training")
        print_gpu_memory_usage(f"Before training '{model_type}' model")

        trainer.train()
        
        print_gpu_memory_usage(f"After training '{model_type}' model")

        adapter_path = os.path.join(output_dir, f"adapter-weights-epoch-{int(self.args.num_train_epochs)}.pt")
        if model_type == "alice":
            self.alice_adapter_path = adapter_path
        else:
            self.bob_adapter_path = adapter_path
        
        # Clean up the model to free memory
        # del self.base_model
        torch.cuda.empty_cache()
        print_gpu_memory_usage(f"After cleaning up fine-tuning model")

    def run_self_play_loop(self):
        logger.info("Starting self-play loop...")
        
        for i in range(self.args.n_iterations):
            logger.info(f"--- Self-Play Iteration {i+1}/{self.args.n_iterations} ---")
            
            new_programs = []
            alice_training_data = []
            bob_training_data = []
            
            random.shuffle(self.programs)
            
            for j, p_original in enumerate(tqdm(self.programs, desc=f"Iteration {i+1}")):
                tqdm.write(f"\n🟩 --- Processing example {j+1}/{len(self.programs)} in Iteration {i+1} ---")
                tqdm.write(f"Original program (P):\n{p_original}")

                # Alice's turn
                q_candidate, x_candidate, alice_raw_output = self.run_alice(p_original)
                
                if not q_candidate:
                    logger.warning("🟨 Alice failed to generate a candidate program.")
                    continue
                
                # Bob's turn
                x_bob, bob_raw_output = self.run_bob(p_original, q_candidate)
                
                is_divergent_alice = self.executor.check_divergence(p_original, q_candidate, x_candidate)
                
                if x_bob: # Bob found a diverging input
                    is_divergent_bob = self.executor.check_divergence(p_original, q_candidate, x_bob)
                    if is_divergent_bob:
                        logger.info(f"🟨 Bob found a valid diverging input. Hard example.")
                        # This is a hard negative for Alice, but we can still use it as a positive example for Bob
                        bob_training_data.append(BOB_PROMPT_TEMPLATE.format(program_p=p_original, program_q=q_candidate) + bob_raw_output)
                        new_programs.append(q_candidate)
                    else:
                        logger.info(f"🟨 Bob's input did not diverge. Easy example.")
                        # This is a case where Bob failed, so it's a negative example for Bob.
                        # We can treat this as a positive example for Alice, as her program was hard enough.
                        if is_divergent_alice:
                            alice_training_data.append(ALICE_PROMPT_TEMPLATE.format(program=p_original) + alice_raw_output)
                            new_programs.append(q_candidate)

                else: # Bob thinks they are equivalent
                    if is_divergent_alice:
                        logger.info("🟨 Bob failed to find a diverging input. Hard example.")
                        # Hard positive for Bob (as it should have found the input)
                        # And a positive for Alice (as it was a good, hard example)
                        bob_completion = f"\n**Analysis:**\n<analysis>\n\n**Diverging Input `x`:**\n```\n{x_candidate}\n```"
                        bob_training_data.append(BOB_PROMPT_TEMPLATE.format(program_p=p_original, program_q=q_candidate) + bob_completion)
                        alice_training_data.append(ALICE_PROMPT_TEMPLATE.format(program=p_original) + alice_raw_output)
                        new_programs.append(q_candidate)
                    else:
                        logger.info("🟨 Both Alice and Bob agree. Easy example.")
                        # Alice's output was not divergent, so it's a negative example for her.
                        # We won't add it to her training data.
                        pass
            
            if alice_training_data or bob_training_data:
                logger.info("Releasing vLLM model to free up memory for fine-tuning...")
                # del self.vllm_model
                torch.cuda.empty_cache()
                print_gpu_memory_usage("After releasing vLLM model")

                if alice_training_data:
                    self.finetune_model(alice_training_data, "alice")
                if bob_training_data:
                    self.finetune_model(bob_training_data, "bob")
                
                logger.info("Re-initializing vLLM model for the next generation round...")
                # self._initialize_vllm()

            
            self.programs.extend(new_programs)

            self.evaluate(i)

    def evaluate(self, iteration):
        logger.info(f"Starting final evaluation for iteration {iteration}...")
        
        # Evaluate Alice's model
        if self.alice_adapter_path:
            self.evaluate_agent("alice", self.alice_adapter_path, iteration)
        else:
            logger.info("Alice adapter not found, skipping evaluation.")

    def evaluate_agent(self, agent_name, adapter_path, iteration):
        logger.info(f"--- Evaluating {agent_name}'s Model for iteration {iteration} ---")
        final_model_path = os.path.join(self.args.output_dir, f"{agent_name}_model_iter_{iteration}")
        
        if not (adapter_path and os.path.exists(adapter_path)):
            logger.warning(f"Adapter path for {agent_name} does not exist: {adapter_path}. Skipping evaluation.")
            return

        logger.info(f"Loading adapter from {adapter_path}")
        
        # Create a new base model instance for merging to avoid conflicts
        logger.info("Loading base model for merging...")
        base_model_for_merging = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="cpu" # Load on CPU to avoid using up GPU memory
        )

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
        )
        peft_model = get_peft_model(base_model_for_merging, lora_config)
        
        adapter_weights = torch.load(adapter_path, map_location='cpu')
        peft_model.load_state_dict(adapter_weights, strict=False)
        
        logger.info(f"Merging adapter for {agent_name}...")
        model_to_save = peft_model.merge_and_unload()
        logger.info(f"Merged adapter for {agent_name}.")

        logger.info(f"Saving final model for {agent_name} to {final_model_path}")
        model_to_save.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Path to the eval script, relative to this file's location
        eval_script_path = 'eval_script/eval_hs_finetuning.sh'
        eval_working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Evaluation/HumanEval'))

        logger.info(f"Submitting evaluation script for {agent_name} via sbatch...")
        subprocess.run(['sbatch', eval_script_path, final_model_path], check=True, cwd=eval_working_dir)
        logger.info(f"Evaluation script for {agent_name} submitted.")


def main():
    parser = argparse.ArgumentParser(description="SInQ Self-Play and Fine-tuning")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--train_filename', type=str, default='data/train.txt', help="Path to the training data.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save models and results.")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p for sampling.")
    parser.add_argument('--top_k', type=int, default=20, help="Top-k for sampling.")
    parser.add_argument('--presence_penalty', type=float, default=1.5, help="Presence penalty for sampling.")
    parser.add_argument('--timeout', type=float, default=10.0, help="Timeout for code execution.")
    parser.add_argument('--n_iterations', type=int, default=1, help="Number of self-play iterations.")
    parser.add_argument('--n_samples', type=int, default=10, help="Number of samples for Alice to generate.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for fine-tuning.")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device during training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument('--lora_r', type=int, default=8, help="LoRA r.")
    parser.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--max_model_len', type=int, default=4096, help="Maximum sequence length for the vLLM model.")
    
    args = parser.parse_args()
    
    sinq = SInQ(args)
    sinq.run_self_play_loop()

if __name__ == "__main__":
    main()



