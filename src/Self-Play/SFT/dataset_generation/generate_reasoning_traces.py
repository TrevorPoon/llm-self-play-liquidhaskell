import os
import re
import random
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from dotenv import load_dotenv
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder
from datasets import load_dataset, load_from_disk
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def load_and_prepare_code_samples(dataset_path: str, data_fraction: float):
    """Loads Haskell code from the specified dataset and samples a fraction."""
    print(f"Loading code samples from '{dataset_path}'...")
    try:
        if os.path.isdir(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            dataset = load_dataset(dataset_path, split="train")
        
        # Ensure 'code' column exists
        if 'code' not in dataset.column_names:
            print(f"Error: 'code' column not found in the dataset at '{dataset_path}'.")
            return []

        num_samples = int(len(dataset) * data_fraction)
        print(f"Original dataset has {len(dataset)} samples. Using {data_fraction*100:.2f}% ({num_samples} samples).")
        
        sampled_dataset = dataset.select(range(num_samples))
        
        return [example['code'] for example in sampled_dataset]
    except Exception as e:
        print(f"Failed to load or process dataset: {e}")
        return []

def build_backward_reasoning_prompt(code: str):
    """
    Constructs a few-shot prompt to guide the model to generate an instruction
    and a reasoning trace for a given piece of Haskell code.
    """
    system_prompt = """
You are an expert Haskell programmer and prompt engineer. Your task is to reverse-engineer a given Haskell code snippet to create a high-quality synthetic data sample for training a reasoning-focused coding model. For the provided Haskell code, you must generate two things:

1.  **A Clear Instruction**: Write a concise and natural-language instruction that a user might have given to request this specific code.
2.  **A Step-by-Step Reasoning Trace**: Provide a detailed, step-by-step thinking process that a developer might follow to arrive at the solution. This should explain the logic behind the implementation and any relevant design choices. Enclose this reasoning in `<think>...</think>` tags.

The output must be structured with the instruction first, then the reasoning trace. Do not repeat the Haskell code in your response.

---
### FEW-SHOT EXAMPLE 1

**Haskell Code:**
```haskell
{-@ sumNats :: [Nat] -> Nat @-}
sumNats :: [Int] -> Int
sumNats [] = 0
sumNats (x:xs) = x + sumNats xs
```

**Your Generated Response:**
**Instruction:**
Write a function that calculates the sum of a list of non-negative integers, including a Liquid Haskell specification to enforce the type constraints.

**Reasoning:**
<think>
The user wants a function to sum a list of non-negative integers and include a Liquid Haskell specification.
1.  **Haskell Function**: I'll define a recursive function `sumNats`. The base case is an empty list, which should return 0. The recursive step is to take the head `x` and add it to the sum of the tail `xs`. The type will be `[Int] -> Int`.
2.  **Liquid Haskell Specification**: The core requirement is that the input list contains only non-negative integers and the output is also non-negative. The `Nat` type alias in Liquid Haskell represents `{v:Int | v >= 0}`.
    - The input type should be `[Nat]` (a list of natural numbers).
    - The output will also be a `Nat`, as the sum of non-negative numbers is always non-negative.
    - This leads to the specification `{-@ sumNats :: [Nat] -> Nat @-}`.
</think>

---
### FEW-SHOT EXAMPLE 2

**Haskell Code:**
```haskell
{-@ safeHead :: {v:[a] | len v > 0} -> a @-}
safeHead :: [a] -> a
safeHead (x:_) = x
```

**Your Generated Response:**
**Instruction:**
Create a safe version of the 'head' function that only accepts non-empty lists, and use a Liquid Haskell specification to guarantee this precondition.

**Reasoning:**
<think>
The goal is to implement a `head` function that is guaranteed to be safe by the type system, meaning it cannot be called with an empty list.
1.  **Haskell Function**: The function `safeHead` will take a list `[a]` and return the first element of type `a`. Since the list is guaranteed to be non-empty, I only need to handle the pattern `(x:_)`, which matches lists with at least one element. I can ignore the `[]` case, as the Liquid Haskell specification will make it unreachable.
2.  **Liquid Haskell Specification**: The precondition is that the input list must not be empty. I can express this with a refinement type: `{v:[a] | len v > 0}`. This type states that the value `v` is a list of type `[a]` whose length is greater than 0. The function returns a value of type `a`.
    - The final specification is `{-@ safeHead :: {v:[a] | len v > 0} -> a @-}`. This ensures that any call to `safeHead` with an empty list will be a compile-time error.
</think>
---
"""
    user_prompt = f"""
Now, complete the following task.

**Haskell Code:**
```haskell
{code}
```

**Your Generated Response:**
"""
    return system_prompt.strip(), user_prompt.strip()

def parse_model_output(completion: str):
    """
    Parses the model's output to extract the instruction and reasoning with improved flexibility.
    It handles variations in headers, ordering, and potential model chatter.
    """
    # Flexible regex for instruction: looks for "**Instruction:**" and captures until
    # the next major section ("**Reasoning:**" or "<think>") or the end of the string.
    # It is case-insensitive and not anchored to the start of the output.
    instruction_regex = r"\*\*Instruction:\*\*\s*(.*?)(?=\s*\*\*Reasoning:\*\*|<think>|$)"
    instruction_match = re.search(instruction_regex, completion, re.DOTALL | re.IGNORECASE)
    instruction = instruction_match.group(1).strip() if instruction_match else ""

    # For reasoning, prioritize the structured <think> tags as they are a clear delimiter.
    reasoning_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    reasoning = ""
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        # As a fallback, look for a "**Reasoning:**" header. This is less structured
        # but provides a good fallback.
        reasoning_regex = r"\*\*Reasoning:\*\*\s*(.*)"
        reasoning_match = re.search(reasoning_regex, completion, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

    return instruction, reasoning

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces for a Haskell dataset using a local VLLM instance.")
    parser.add_argument('--source_dataset_path', type=str, required=True, help="Path to the source Haskell dataset (local path or HF name).")
    parser.add_argument('--data_fraction', type=float, default=0.3, help="Fraction of the source dataset to use for generation.")
    parser.add_argument('--output_dir', type=str, default='../data/synthetic_reasoning_dataset_raw', help="Directory to save the generated dataset.")
    parser.add_argument('--output_filename_arrow', type=str, default='synthetic_reasoning_dataset.arrow', help="Filename for the output Arrow dataset.")
    parser.add_argument('--output_filename_jsonl', type=str, default='synthetic_reasoning_dataset.jsonl', help="Filename for the output JSONL dataset.")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", help="The name of the model to use for generation (via VLLM).")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of new tokens to generate per sample.")
    parser.add_argument('--max_model_len', type=int, default=8192, help="Maximum sequence length for the model.")
    parser.add_argument('--quantization', type=str, default=None, help="Quantization method to use (e.g., 'awq', 'bnb').")
    parser.add_argument('--dtype', type=str, default='bfloat16', help="Data type to use for the model (e.g., 'bfloat16').")
    parser.add_argument('--pipeline_parallel_size', type=int, default=1, help="Number of pipeline parallelism stages (GPUs). Set to 1 to disable.")
    
    args = parser.parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # --- 1. Configure VLLM ---
    print(f"Loading model {args.model} with vLLM...")
    num_gpus = torch.cuda.device_count()
    
    # If quantization is enabled, disable tensor parallelism to avoid ValueError with BitsAndBytes.
    # If pipeline_parallel_size is explicitly set, use it. Otherwise, default to num_gpus for pipeline.
    if args.quantization:
        print(f"Quantization '{args.quantization}' enabled. Forcing tensor_parallel_size=1.")
        tensor_parallel_size = 1
        if args.pipeline_parallel_size == 1 and num_gpus > 1: # If default and multiple GPUs, suggest using all for pipeline
            print(f"Suggestion: With quantization, consider setting --pipeline_parallel_size={num_gpus} to utilize all GPUs.")
    else:
        # If no quantization, default to tensor parallelism across all GPUs for efficiency
        # unless pipeline_parallel_size is explicitly set.
        if args.pipeline_parallel_size > 1:
            print(f"Pipeline parallelism requested (--pipeline_parallel_size={args.pipeline_parallel_size}). Setting tensor_parallel_size=1.")
            tensor_parallel_size = 1
        else:
            print(f"No quantization or pipeline parallelism specified. Defaulting to tensor_parallel_size={num_gpus}.")
            tensor_parallel_size = num_gpus

    # Ensure pipeline_parallel_size is set based on explicit argument, or default to 1 if not used.
    pipeline_parallel_size = args.pipeline_parallel_size

    print(f"Initializing vLLM with tensor_parallel_size={tensor_parallel_size} and pipeline_parallel_size={pipeline_parallel_size}")
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_new_tokens,
        presence_penalty=1.5,
    )
    
    # --- 2. Load and Prepare Code Samples ---
    code_samples = load_and_prepare_code_samples(args.source_dataset_path, args.data_fraction)
    if not code_samples:
        print("No code samples were loaded. Exiting.")
        return

    # --- 3. Generate Data ---
    print(f"Generating reasoning traces for {len(code_samples)} samples...")
    dataset = []

    # Batch prepare prompts
    full_prompts = []
    original_code_snippets = []
    for code in tqdm(code_samples, desc="Preparing Prompts"):
        system_prompt, user_prompt = build_backward_reasoning_prompt(code)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompts.append(full_prompt)
        original_code_snippets.append(code)

    # Generate in a single batch
    print("Generating completions with vLLM...")
    vllm_outputs = llm.generate(full_prompts, sampling_params)
    print("Generation complete.")
    
    # Process outputs
    for i, output in tqdm(enumerate(vllm_outputs), total=len(vllm_outputs), desc="Processing Outputs"):
        completion = output.outputs[0].text
        original_code = original_code_snippets[i]

        instruction, reasoning = parse_model_output(completion)

        if instruction and reasoning:
            dataset.append({
                "instruction": instruction,
                "reasoning": reasoning,
                "code": original_code,
            })
        else:
            print(f"\n--- WARNING: Failed to parse output for sample {i+1} ---")
            print(f"Original Code:\n{original_code}")
            print(f"Model Completion:\n{completion}")
            print("----------------------------------------------------")


    if not dataset:
        print("No valid data was generated. Exiting.")
        return
        
    # --- 4. Save Generated Dataset ---
    if dataset:
        print(f"Saving generated dataset to '{args.output_dir}'...")
        # Convert list of dicts to Hugging Face Dataset
        from datasets import Dataset
        hf_dataset = Dataset.from_list(dataset)
        
        os.makedirs(args.output_dir, exist_ok=True)
        hf_dataset.save_to_disk(args.output_dir)
        print("Generated dataset saved successfully.")
    else:
        print("No dataset generated to save.")

if __name__ == "__main__":
    main() 