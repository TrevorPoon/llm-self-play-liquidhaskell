# Script for testing the adapter
import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained LoRA adapter.")
    
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Path to the base model.")
    parser.add_argument('--adapter_path', type=str, required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument('--prompt', type=str, required=True, help="The instruction prompt to generate code from.")
    
    parser.add_argument('--max_new_tokens', type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.1, help="Temperature for sampling.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p for sampling.")

    args = parser.parse_args()

    # --- Load Model and Tokenizer ---
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # --- Load LoRA Adapter ---
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    # --- Merge and Unload for Faster Inference ---
    print("Merging adapter weights...")
    model = model.merge_and_unload()
    
    print("Model ready for inference.")

    # --- Create Formatted Prompt ---
    # Using the same format as in data preparation
    formatted_prompt = f"""You are an expert Haskell programmer. Your task is to write a correct and efficient Haskell function that solves the given problem. Please provide a type signature for your function.

### Instruction:
{args.prompt}

### Response:
"""

    # --- Generate Code ---
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("\n--- Generating Response ---")
    sequences = pipe(
        formatted_prompt,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
    )
    
    print("\n--- Generated Haskell Code ---")
    print(sequences[0]['generated_text'])

if __name__ == "__main__":
    main() 