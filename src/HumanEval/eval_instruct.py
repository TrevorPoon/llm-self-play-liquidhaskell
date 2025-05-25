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
        <think>
        """


def generate_one(example, lang, tokenizer, model, max_new_tokens):
    prompt = build_deepseekcoder_instruction(language_settings[lang]['full_name'], example['prompt'])
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    stop_id = tokenizer.eos_token_id # Use the tokenizer's own EOS token ID
    # Ensure stop_id is an int, otherwise generation will fail.
    if isinstance(stop_id, list): # Sometimes it can be a list
        stop_id = stop_id[0]
    assert isinstance(stop_id, int), f"Invalid tokenizer, EOS id not found or not an int: {stop_id}"

    with torch.inference_mode():
        with torch.autocast(device_type=model.device.type, dtype=torch.float16):
            outputs = model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs), # Explicitly pass attention_mask
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # top_p=0.95,
                # temperature=temperature,
                pad_token_id=stop_id, # It's common to use EOS as PAD when no other pad_token is set
                eos_token_id=stop_id
            )

    output = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    example['output'] = output
    
    # Return both the example and the success status
    return extract_generation_code(example, lang_code=lang)

def generate_main(args):
    model_name_or_path = args.model
    lang = args.language
    saved_path = args.output_path
    temp_dir = args.temp_dir
    max_new_tokens = args.max_new_tokens
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")

    print("model", model_name_or_path)
    print("lang", lang)
    print("max_new_tokens", max_new_tokens)
    
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
    print("Read {} examples for evaluation over.".format(len(examples)))

    generated_examples = []
    failed_extraction_count = 0  # Initialize counter
    for ex in tqdm(examples, desc='Generating'):
        gen_example, extraction_successful = generate_one(ex, args.language, tokenizer, model, max_new_tokens)
        if not extraction_successful:
            failed_extraction_count += 1
        generated_examples.append(gen_example)

    print("Generate all over!!!")

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(saved_path)
    if output_dir: # Check if dirname is not empty (e.g. for relative paths in current dir)
        os.makedirs(output_dir, exist_ok=True)

    with open(saved_path, 'w', encoding='utf-8') as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(generated_examples), saved_path))
    
    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang
    )
    print(lang, result, model_name_or_path)
    print(f"Total failed code extractions: {failed_extraction_count}") # Print the count
    pass

def evaluation_only(args):
    lang = args.language
    temp_dir = args.temp_dir
    assert os.path.exists(args.output_path), "Not fond output file: {}".format(args.output_path)
    os.makedirs(temp_dir, exist_ok=True)

    output_name = os.path.basename(args.output_path)
    output_examples = [json.loads(x) for x in open(args.output_path) if x.strip()]

    processed_examples = []
    # Loop and unpack the tuple, though extraction_successful is not directly used here
    for ex in tqdm(output_examples, "Processing"):
        processed_ex, _ = extract_generation_code(ex, lang)
        processed_examples.append(processed_ex)
        
    processed_path = os.path.join(temp_dir, output_name)
    with open(processed_path, 'w', encoding='utf-8') as fw:
        for ex in processed_examples:
            fw.write(json.dumps(ex) + '\n')
        print("Save {} processed examples into {} over!".format(len(processed_examples), processed_path))

    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    from human_eval.evaluation import evaluate_functional_correctness
    result = evaluate_functional_correctness(
        input_file=processed_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang
    )
    print(lang, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path")
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--language', type=str, help="langauge")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    parser.add_argument('--max_new_tokens', type=int, help="max new tokens", default=4096)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    generate_main(args)
    pass
