import os
import re
import random
import argparse
import time
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder
from datasets import load_dataset


def load_and_prepare_prompts(num_samples: int):
    """Loads prompts from sdiazlor/python-reasoning-dataset, replaces 'Python' with 'Haskell'."""
    print("Loading and preparing prompts from Hugging Face dataset...")
    try:
        dataset = load_dataset("sdiazlor/python-reasoning-dataset", split="train")
        prompts = dataset['prompt']
        print("Replacing 'Python' with 'Haskell' in prompts...")
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
    # Load .env file from the root of the project
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(dotenv_path)

    parser = argparse.ArgumentParser(description="Generate a synthetic Haskell dataset using the OpenRouter API.")
    parser.add_argument('--num_samples', type=int, default=500, help="Number of samples to generate.")
    parser.add_argument('--max_retries', type=int, default=3, help="Maximum number of retries for a failed API call.")
    parser.add_argument('--output_dir', type=str, default='../data', help="Directory to save the dataset.")
    parser.add_argument('--output_filename_arrow', type=str, default='synthetic_liquid_haskell_dataset_openrouter.arrow', help="Filename for the output Arrow dataset.")
    parser.add_argument('--output_filename_jsonl', type=str, default='synthetic_liquid_haskell_dataset_openrouter.jsonl', help="Filename for the output JSONL dataset.")
    parser.add_argument('--upload_to_hf', action='store_true', help="Flag to upload the dataset to Hugging Face Hub.")
    parser.add_argument('--hf_repo_name', type=str, default="synthetic-liquid-haskell-dataset-openrouter", help="The name of the Hugging Face repository.")
    parser.add_argument('--hf_username', type=str, help="Your Hugging Face username (required if uploading).")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", help="The name of the model on OpenRouter to use for generation.")
    parser.add_argument('--openrouter_api_key', type=str, default=None, help="OpenRouter API key. Can also be set via OPENROUTER_API_KEY env var.")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of tokens for the completion.")
    
    args = parser.parse_args()

    if args.upload_to_hf and not args.hf_username:
        parser.error("--hf_username is required when --upload_to_hf is set.")

    # --- 1. Configure OpenRouter Client ---
    api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("OpenRouter API key must be provided via --openrouter_api_key or set as OPENROUTER_API_KEY in a .env file.")

    print(f"Connecting to OpenRouter with model {args.model}...")
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    # --- 2. Load and Prepare Prompts ---
    prompts_for_generation = load_and_prepare_prompts(args.num_samples)
    if not prompts_for_generation:
        print("No prompts were loaded. Exiting.")
        return

    # --- 3. Generate Data ---
    print(f"Generating up to {len(prompts_for_generation)} samples with {args.max_retries} retries each...")
    dataset = []
    
    for problem in tqdm(prompts_for_generation, desc="Generating Liquid Haskell Code via OpenRouter"):
        system_prompt, user_prompt = build_few_shot_liquid_haskell_prompt(problem)
        
        for attempt in range(args.max_retries):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=args.max_new_tokens,
                    temperature=0.6,
                    top_p=0.95,
                )
                
                completion = response.choices[0].message.content

                print(f"\nProblem: {problem}")
                print(f"Completion: {completion}")
                
                if completion:
                    dataset.append({
                        "prompt": problem,
                        "completion": completion,
                    })
                break

            except Exception as e:
                print(f"Attempt {attempt + 1}/{args.max_retries} failed. Error: {e}")
                if attempt < args.max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print("Max retries reached. Skipping this prompt.")

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