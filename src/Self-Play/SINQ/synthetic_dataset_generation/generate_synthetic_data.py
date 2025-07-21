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
from transformers.generation.utils import GenerationMixin
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



def load_and_prepare_prompts(num_samples: int):
    """Loads prompts from nvidia/OpenCodeInstruct, filters, sorts, and replaces 'Python' with 'Haskell'."""
    print("Loading and preparing prompts from Hugging Face dataset (nvidia/OpenCodeInstruct)...")
    try:
        dataset = load_dataset("nvidia/OpenCodeInstruct", split="train")
        
        # =========================================
        # Filter the dataset
        print("Filtering dataset by domain='generic' and generation_algorithm='self-instruct'...")
        filtered_dataset = dataset.filter(
            lambda x: x['domain'] == 'generic' and x['generation_algorithm'] == 'self-instruct'
        )
        print(f"Dataset filtered. Original size: {len(dataset)}, Filtered size: {len(filtered_dataset)}")

        # Add a temporary column for the length of 'output'
        print("Calculating length of 'output' for sorting...")
        filtered_dataset = filtered_dataset.map(lambda x: {'output_length': len(x['output'])})

        # Sort the dataset by 'output_length' (shortest first) then by 'average_test_score' (descending)
        print("Sorting dataset by 'output_length' (ascending) and then by 'average_test_score' (descending)...")
        dataset = filtered_dataset.sort(["output_length", "average_test_score"], reverse=[False, True])
        print("Dataset sorted.")
        # =========================================

        # Get input, output, and unit_tests
        prompts_data = dataset.select_columns(['input', 'output']).select(range(len(dataset)))

        print(f"Loaded {len(prompts_data)} prompts after filtering and sorting.")
        # No need to modify prompts for 'Haskell' here as we want the original Python context
        # The modification will happen in build_few_shot_haskell_prompt
        # random.shuffle(prompts_data) # We want to keep the sorting order
        num_to_take = min(num_samples, len(prompts_data))
        print(f"Loaded {len(prompts_data)} entries, will use {num_to_take} for generation.")

        print(f"First 5 Prompts data: \n{prompts_data[:5]}")

        return prompts_data.select(range(num_to_take))
    
    except Exception as e:
        print(f"Failed to load or process dataset: {e}")
        return []

def build_few_shot_haskell_prompt(problem_idea: str, python_code: str):
    """
    Constructs a detailed few-shot prompt to guide the model to generate
    Haskell code from a given Python problem, code, and unit tests.
    """
    system_prompt = """
You are an expert in Haskell programming. Your goal is to generate simple, self-contained Haskell code examples that are functionally equivalent to given Python code, and satisfy the given unit tests.

Follow the format of these examples precisely. For each task, you must:
1.  **Provide the Solution**: Provide the complete, standalone Haskell code. The code must be enclosed in a ```haskell ... ``` block.
2.  **Keep it Simple**: The code should be very basic and easy to understand for a beginner.

---
### FEW-SHOT EXAMPLE 1

**Instruction:**
Write a function that calculates the sum of a list of non-negative integers.

**Python Code:**
```python
def sum_non_negative_list(nums):
    total = 0
    for num in nums:
        total += num
    return total
```

**Python Unit Tests:**
```python
assert sum_non_negative_list([1, 2, 3]) == 6
assert sum_non_negative_list([]) == 0
assert sum_non_negative_list([0]) == 0
assert sum_non_negative_list([10, 20]) == 30
```

**Response:**
```haskell
sumNats :: [Int] -> Int
sumNats [] = 0
sumNats (x:xs) = x + sumNats xs
```

---
### FEW-SHOT EXAMPLE 2

**Instruction:**
Write a function that returns the square of a given number.

**Python Code:**
```python
def square(x):
    return x * x
```

**Response:**
```haskell
square :: Int -> Int
square x = x * x
```
---
"""
    user_prompt = f"""
Now, complete the following task.

**Instruction:**
{problem_idea}

**Python Code:**
```python
{python_code}
```

**Response:**
"""
    return system_prompt.strip(), user_prompt.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic Haskell dataset using a local VLLM instance.")
    parser.add_argument('--num_samples', type=int, default=500, help="Number of samples to generate.")
    parser.add_argument('--output_dir', type=str, default='../data', help="Directory to save the dataset.")
    parser.add_argument('--output_filename_arrow', type=str, default='synthetic_haskell_dataset_OpenCode_Instruct.arrow', help="Filename for the output Arrow dataset.")
    parser.add_argument('--output_filename_jsonl', type=str, default='synthetic_haskell_dataset_OpenCode_Instruct.jsonl', help="Filename for the output JSONL dataset.")
    parser.add_argument('--upload_to_hf', action='store_true', help="Flag to upload the dataset to Hugging Face Hub.")
    parser.add_argument('--hf_repo_name', type=str, default="synthetic-haskell-dataset", help="The name of the Hugging Face repository.")
    parser.add_argument('--hf_username', type=str, help="Your Hugging Face username (required if uploading).")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", help="The name of the model to use for generation (via VLLM).")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of new tokens to generate per sample.")
    parser.add_argument('--max_model_len', type=int, default=8192, help="Maximum sequence length for the model.")
    parser.add_argument('--dtype', type=str, default='bfloat16', help="Data type to use for the model (e.g., 'bfloat16').") # bfloat16, fp8, fp16, fp32
    
    args = parser.parse_args()

    if args.upload_to_hf and not args.hf_username:
        parser.error("--hf_username is required when --upload_to_hf is set.")

    # --- 1. Configure VLLM ---
    print(f"Loading model {args.model} with vLLM...")
    num_gpus = torch.cuda.device_count()
    
    tensor_parallel_size = num_gpus

    print(f"Initializing vLLM with tensor_parallel_size={tensor_parallel_size}")
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_new_tokens,
        presence_penalty=1.5,
    )
    
    # --- 2. Load and Prepare Prompts ---
    print(f"Loading and preparing {args.num_samples} prompts...")
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
        system_prompt, user_prompt = build_few_shot_haskell_prompt(problem['input'], problem['output'])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompts.append(full_prompt)
        original_prompts.append(problem['input']) # Store the original problem statement

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
        
        # Extract Haskell code from completion
        haskell_code_match = re.search(r'''```haskell\n(.*?)```''', completion, re.DOTALL)
        
        if haskell_code_match:
            haskell_code = haskell_code_match.group(1).strip()
            dataset.append({
                "prompt": problem,
                "code": haskell_code,
            })
        else:
            print(f"No Haskell code found in completion for problem")

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