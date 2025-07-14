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
from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_and_prepare_prompts(num_samples: int):
    """Loads prompts from nvidia/OpenCodeInstruct, filters, sorts, and replaces 'Python' with 'Haskell'."""
    print("Loading and preparing prompts from Hugging Face dataset (nvidia/OpenCodeInstruct)...")
    try:
        dataset = load_dataset("nvidia/OpenCodeInstruct", split="train")

        # Filter the dataset
        print("Filtering dataset by domain='generic' and generation_algorithm='self-instruct'...")
        filtered_dataset = dataset.filter(
            lambda x: x['domain'] == 'generic' and x['generation_algorithm'] == 'self-instruct'
        )
        print(f"Dataset filtered. Original size: {len(dataset)}, Filtered size: {len(filtered_dataset)}")

        # Sort the dataset
        print("Sorting dataset by 'average_test_score' in descending order...")
        sorted_dataset = filtered_dataset.sort("average_test_score", reverse=True)
        print("Dataset sorted.")

        # Get the first 100,000 samples or fewer if not enough
        num_to_take_from_filtered = min(100000, len(sorted_dataset))
        prompts = sorted_dataset['input'][:num_to_take_from_filtered]

        print(f"First 100 prompts: {prompts[:100]}")

        print(f"Loaded {len(prompts)} prompts after filtering and sorting.")
        modified_prompts = [re.sub(r'python', 'Haskell', p, flags=re.IGNORECASE) for p in prompts]
        random.shuffle(modified_prompts)
        num_to_take = min(num_samples, len(modified_prompts))
        print(f"Loaded {len(modified_prompts)} prompts, will use {num_to_take} for generation.")
        return modified_prompts[:num_to_take]
    except Exception as e:
        print(f"Failed to load or process dataset: {e}")
        return []


def build_few_shot_liquid_haskell_prompt(problem_idea: str):
    """
    Constructs a detailed few-shot prompt to guide the model to generate
    Haskell code with Liquid Haskell specifications.
    """
    system_prompt = """
You are an expert in Haskell and Liquid Haskell. Your goal is to generate simple, self-contained Haskell code examples along with their corresponding Liquid Haskell specifications. Liquid Haskell uses refinement types to add formal, machine-checkable proofs about Haskell programs.

Follow the format of these examples precisely. For each task, you must:
1.  **Think Step-by-Step**: Provide your reasoning process in `<think>...</think>` tags. Explain the logic for both the Haskell function and its Liquid Haskell specification.
2.  **Provide the Solution**: Provide the complete, standalone Haskell code, including the `{-@ ... @-}` block for the Liquid Haskell specification. The code must be enclosed in a ```haskell ... ``` block.
3.  **Keep it Simple**: The code should be very basic and easy to understand for a beginner.

---
### FEW-SHOT EXAMPLE 1

**Instruction:**
Write a function that calculates the sum of a list of non-negative integers.

**Response:**
<think>
The user wants a function to sum a list of non-negative integers.
1.  **Haskell Function**: I'll define a function `sumNats` that takes a list of `Int` and returns an `Int`. I can implement this recursively. The base case is an empty list, which sums to 0. The recursive step is to add the head of the list to the sum of the tail.
2.  **Liquid Haskell Specification**: The key constraint is that the input list contains only "non-negative integers". In Liquid Haskell, this is represented by the `Nat` type alias, which stands for `{v:Int | v >= 0}`.
    - The input type should be `[Nat]`, a list of natural numbers.
    - The output will also be a natural number, since summing non-negative numbers results in a non-negative number. So the return type is `Nat`.
    - The final specification will be `{-@ sumNats :: [Nat] -> Nat @-}`.
</think>
```haskell
{-@ sumNats :: [Nat] -> Nat @-}
sumNats :: [Int] -> Int
sumNats [] = 0
sumNats (x:xs) = x + sumNats xs
```

---
### FEW-SHOT EXAMPLE 2

**Instruction:**
Write a function that safely returns the first element of a non-empty list.

**Response:**
<think>
The user wants a "safe" head function that only operates on non-empty lists. This implies a partial function, where the caller must guarantee the precondition.
1.  **Haskell Function**: The function `safeHead` will take a list `[a]` and return an element of type `a`. Since it's for non-empty lists, I only need to handle the `(x:_)` pattern, where `x` is the head. I don't need a case for `[]` because the type signature will prevent it.
2.  **Liquid Haskell Specification**: The precondition is that the list is "non-empty".
    - I can express this using a refinement on the input list type: `{v:[a] | len v > 0}`. This means "a list `v` of type `[a]` such that its length is greater than 0".
    - The function returns an element of type `a`.
    - The final specification will be `{-@ safeHead :: {v:[a] | len v > 0} -> a @-}`. This contract ensures that anyone calling `safeHead` must provide a non-empty list, making the function safe at compile-time with Liquid Haskell.
</think>
```haskell
{-@ safeHead :: {v:[a] | len v > 0} -> a @-}
safeHead :: [a] -> a
safeHead (x:_) = x
```
---
"""
    user_prompt = f"""
Now, complete the following task.

**Instruction:**
{problem_idea}

**Response:**
"""
    return system_prompt.strip(), user_prompt.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic Haskell dataset using a local VLLM instance.")
    parser.add_argument('--num_samples', type=int, default=500, help="Number of samples to generate.")
    parser.add_argument('--output_dir', type=str, default='../data', help="Directory to save the dataset.")
    parser.add_argument('--output_filename_arrow', type=str, default='synthetic_liquid_haskell_dataset_OpenCode_Instruct.arrow', help="Filename for the output Arrow dataset.")
    parser.add_argument('--output_filename_jsonl', type=str, default='synthetic_liquid_haskell_dataset_OpenCode_Instruct.jsonl', help="Filename for the output JSONL dataset.")
    parser.add_argument('--upload_to_hf', action='store_true', help="Flag to upload the dataset to Hugging Face Hub.")
    parser.add_argument('--hf_repo_name', type=str, default="synthetic-liquid-haskell-dataset", help="The name of the Hugging Face repository.")
    parser.add_argument('--hf_username', type=str, help="Your Hugging Face username (required if uploading).")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", help="The name of the model to use for generation (via VLLM).")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of new tokens to generate per sample.")
    parser.add_argument('--max_model_len', type=int, default=8192, help="Maximum sequence length for the model.")
    parser.add_argument('--quantization', type=str, default=None, help="Quantization method to use (e.g., 'awq', 'bnb').")
    parser.add_argument('--dtype', type=str, default='bfloat16', help="Data type to use for the model (e.g., 'bfloat16').")
    parser.add_argument('--pipeline_parallel_size', type=int, default=1, help="Number of pipeline parallelism stages (GPUs). Set to 1 to disable.")
    
    args = parser.parse_args()

    if args.upload_to_hf and not args.hf_username:
        parser.error("--hf_username is required when --upload_to_hf is set.")

    # --- 1. Configure VLLM ---
    print(f"Loading model {args.model} with vLLM...")
    num_gpus = torch.cuda.device_count()
    
    # If quantization is enabled, disable tensor parallelism to avoid ValueError with BitsAndBytes.
    # If pipeline_parallel_size is explicitly set, use it. Otherwise, default to num_gpus for pipeline.
    if args.quantization:
        print(f"Quantization '{args.quantization}' enabled. Forcing tensor_parallel_size=1.")
        tensor_parallel_size = 1
        if args.pipeline_parallel_size == 1 and num_gpus > 1: # If default and multiple GPUs, suggest using all for pipeline
            print(f"Suggestion: With quantization, consider setting --pipeline_parallel_size={{num_gpus}} to utilize all GPUs.")
    else:
        # If no quantization, default to tensor parallelism across all GPUs for efficiency
        # unless pipeline_parallel_size is explicitly set.
        if args.pipeline_parallel_size > 1:
            print(f"Pipeline parallelism requested (--pipeline_parallel_size={{args.pipeline_parallel_size}}). Setting tensor_parallel_size=1.")
            tensor_parallel_size = 1
        else:
            print(f"No quantization or pipeline parallelism specified. Defaulting to tensor_parallel_size={{num_gpus}}.")
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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
    )
    
    # --- 2. Load and Prepare Prompts ---
    prompts_for_generation = load_and_prepare_prompts(args.num_samples)
    if not prompts_for_generation:
        print("No prompts were loaded. Exiting.")
        return

    # --- 3. Generate Data ---
    print(f"Generating up to {len(prompts_for_generation)} samples...")
    dataset = []

    # Batch prepare prompts
    full_prompts = []
    original_prompts = []  # To store the original problem statement
    for problem in tqdm(prompts_for_generation, desc="Preparing Prompts"):
        system_prompt, user_prompt = build_few_shot_liquid_haskell_prompt(problem)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompts.append(full_prompt)
        original_prompts.append(problem)

    # Generate in a single batch
    print("Generating completions with vLLM...")
    vllm_outputs = llm.generate(full_prompts, sampling_params)
    print("Generation complete.")
    
    # Process outputs
    for i, output in tqdm(enumerate(vllm_outputs), total=len(vllm_outputs), desc="Processing Outputs"):
        completion = output.outputs[0].text
        problem = original_prompts[i]

        print(f"\nProblem: {problem}")
        print(f"Completion: {completion}")
        
        if completion:
            dataset.append({
                "prompt": problem,
                "completion": completion,
            })

    if not dataset:
        print("No data was generated. Exiting.")
        return
        
    # --- 4. Save Dataset to Arrow and JSONL files ---
    print(f"Generated {len(dataset)} valid samples.")
    df = pd.DataFrame(dataset)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path_arrow = os.path.join(args.output_dir, args.output_filename_arrow)
    print(f"Saving dataset to {output_path_arrow}...")
    table = pa.Table.from_pandas(df)
    feather.write_feather(table, output_path_arrow)
    print("Arrow dataset saved successfully.")

    output_path_jsonl = os.path.join(args.output_dir, args.output_filename_jsonl)
    print(f"Saving dataset to {output_path_jsonl}...")
    df.to_json(output_path_jsonl, orient='records', lines=True)
    print("JSONL dataset saved successfully.")


    # --- 5. Upload to Hugging Face Hub ---
    if args.upload_to_hf:
        print(f"Uploading Arrow dataset to Hugging Face Hub: {args.hf_username}/{args.hf_repo_name}")
        
        if not HfFolder.get_token():
            print("Hugging Face token not found. Please log in using `huggingface-cli login`.")
            return

        api = HfApi()
        repo_id = f"{args.hf_username}/{args.hf_repo_name}"
        
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=output_path_arrow,
            path_in_repo=args.output_filename_arrow,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Dataset successfully uploaded to {repo_id}.")


if __name__ == "__main__":
    main() 