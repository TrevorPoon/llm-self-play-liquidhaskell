import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm

data_abs_dir = Path(__file__).parent / "data"

from utils.utils import extract_generation_code, language_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness

def build_deepseekcoder_instruction(language: str, question: str):
    # New instruction format incorporating the <think> block
    return f"""Below is an instruction that describes a task.
        Write a response that appropriately completes the request.
        Before generating the code, think carefully about the problem and write down a step-by-step plan in the <think> block.
        Then, complete the function, returning all completed function in a markdown codeblock. You are not allowed to modify the given code outside the completion.

        ### Instruction:
        Complete the following {language} function:

        ### Given Code:
        ```{language.lower()}
        {question.strip()}
        ```

        ### Response:
        """.strip()


def generate_one(example, lang, tokenizer, model, max_new_tokens):
    prompt = build_deepseekcoder_instruction(language_settings[lang]['full_name'], example['prompt'])
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.eos_token_id # Use the tokenizer's own EOS token ID
    print(f"[DEBUG][generate_one] Stop ID: {stop_id}")
    # Ensure stop_id is an int, otherwise generation will fail.
    if isinstance(stop_id, list): # Sometimes it can be a list
        stop_id = stop_id[0]
    assert isinstance(stop_id, int), f"Invalid tokenizer, EOS id not found or not an int: {stop_id}"

    print(f"\n[DEBUG][generate_one] Task ID: {example.get('task_id', 'N/A')}")
    print(f"[DEBUG][generate_one] Prompt sent to model:\n{prompt}")

    with torch.inference_mode():
        with torch.autocast(device_type=model.device.type, dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[-1] + max_new_tokens,
                do_sample=False,
                # top_p=0.95,
                # temperature=temperature,
                pad_token_id=stop_id, # It's common to use EOS as PAD when no other pad_token is set
                eos_token_id=stop_id
            )

    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"[DEBUG][generate_one] Raw model output for Task ID {example.get('task_id', 'N/A')}:\n{output}")
    example['output'] = output
    
    # Return both the example and the success status
    gen_example, success = extract_generation_code(example, lang_code=lang)
    print(f"[DEBUG][generate_one] Extraction success for Task ID {example.get('task_id', 'N/A')}: {success}")
    return gen_example, success

def generate_main(args):
    model_name_or_path = args.model
    lang = args.language
    saved_path = args.output_path
    temp_dir = args.temp_dir
    max_new_tokens = args.max_new_tokens
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")

    print(f"[DEBUG][generate_main] model: {model_name_or_path}")
    print(f"[DEBUG][generate_main] lang: {lang}")
    print(f"[DEBUG][generate_main] max_new_tokens: {max_new_tokens}")
    print(f"[DEBUG][generate_main] Problem file: {problem_file}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        #use_flash_attention_2=True
    )
    model = torch.compile(model)
    model.eval()
    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    
    # For debugging purposes, only use the first 5 questions. Comment out the line below to run on all questions.
    examples = examples[:10]
    
    print(f"Read {len(examples)} examples for evaluation over.")

    generated_examples = []
    failed_extraction_count = 0  # Initialize counter
    for ex in tqdm(examples, desc='Generating'):
        print(f"\n[DEBUG][generate_main] Processing Task ID: {ex.get('task_id', 'N/A')}")
        gen_example, extraction_successful = generate_one(ex, args.language, tokenizer, model, max_new_tokens)
        if not extraction_successful:
            failed_extraction_count += 1
        generated_examples.append(gen_example)

    print("Generate all over!!!")
    print(f"[DEBUG][generate_main] Total failed code extractions: {failed_extraction_count}")

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(saved_path)
    if output_dir: # Check if dirname is not empty (e.g. for relative paths in current dir)
        os.makedirs(output_dir, exist_ok=True)

    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print(f"Save {len(generated_examples)} processed examples into {saved_path} over!")
    
    print(f"[DEBUG][generate_main] Starting evaluation for {lang}...")
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=120.0,
        problem_file=problem_file,
        language=lang
    )
    print(f"\n[DEBUG][generate_main] Final evaluation result for {lang} on {model_name_or_path}: {result}")
    print(f"Total failed code extractions: {failed_extraction_count}") # Print the count
    pass

def evaluation_only(args):
    lang = args.language
    temp_dir = args.temp_dir
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl") # Added problem_file definition

    print(f"\n[DEBUG][evaluation_only] Language: {lang}")
    print(f"[DEBUG][evaluation_only] Temporary directory: {temp_dir}")
    print(f"[DEBUG][evaluation_only] Output path (input for evaluation): {args.output_path}")
    print(f"[DEBUG][evaluation_only] Problem file: {problem_file}")

    assert os.path.exists(args.output_path), "Not fond output file: {}".format(args.output_path)
    os.makedirs(temp_dir, exist_ok=True)

    output_name = os.path.basename(args.output_path)
    output_examples = [json.loads(x) for x in open(args.output_path) if x.strip()]
    print(f"[DEBUG][evaluation_only] Read {len(output_examples)} examples from {args.output_path}")

    processed_examples = []
    # Loop and unpack the tuple, though extraction_successful is not directly used here
    for ex in tqdm(output_examples, "Processing"):
        print(f"\n[DEBUG][evaluation_only] Processing Task ID: {ex.get('task_id', 'N/A')} from input file.")
        processed_ex, extraction_status = extract_generation_code(ex, lang) # Keep both return values
        print(f"[DEBUG][evaluation_only] Extracted code for Task ID {ex.get('task_id', 'N/A')}:\n{processed_ex.get('generation', 'N/A')}")
        print(f"[DEBUG][evaluation_only] Extraction status for Task ID {ex.get('task_id', 'N/A')}: {extraction_status}")
        processed_examples.append(processed_ex)
        
    processed_path = os.path.join(temp_dir, output_name)
    with open(processed_path, 'w', encoding='utf-8') as fw:
        for ex in processed_examples:
            fw.write(json.dumps(ex) + '\n')
        print(f"Save {len(processed_examples)} processed examples into {processed_path} over!")

    print(f"[DEBUG][evaluation_only] Starting evaluation for {lang}...")
    result = evaluate_functional_correctness(
        input_file=processed_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=120.0,
        problem_file=problem_file,
        language=lang
    )
    print(f"\n[DEBUG][evaluation_only] Final evaluation result for {lang}: {result}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--language', type=str, help="langauge")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    parser.add_argument('--max_new_tokens', type=int, help="max new tokens", default=4096)
    parser.add_argument('--evaluation_only', action='store_true', help="if only evaluate the output file")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.evaluation_only:
        evaluation_only(args)
    else:
        generate_main(args)
    pass
