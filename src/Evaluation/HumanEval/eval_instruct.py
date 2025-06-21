import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from datetime import datetime # Import datetime for timestamping results

from vllm import LLM, SamplingParams

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

data_abs_dir = Path(__file__).parent / "data"

from utils.utils import extract_generation_code, language_settings
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from human_eval.evaluation import evaluate_functional_correctness


class EndThinkBlockProcessor(LogitsProcessor):
    """
    A LogitsProcessor that encourages the model to generate the `</think>` token 
    once the thought process has become too long.
    """
    def __init__(self, 
                 tokenizer, 
                 prompt_len: int, 
                 max_think_length: int = 256, 
                 end_think_bias: float = 5.0):
        """
        Args:
            tokenizer: The model's tokenizer.
            prompt_len (int): The length of the initial prompt tokens.
            max_think_length (int): The number of generated tokens after which to start
                                    encouraging the end of the think block.
            end_think_bias (float): The positive bias to add to the `</think>` token's logit.
                                   A higher value makes generation more likely.
        """
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.max_think_length = max_think_length
        self.end_think_bias = end_think_bias

        # Get token IDs. Handle cases where they might not exist.
        self.think_token_id = self._get_token_id("<think>")
        self.end_think_token_id = self._get_token_id("</think>")

        if self.think_token_id is None or self.end_think_token_id is None:
            print("[WARN] <think> or </think> tokens not found in tokenizer. Processor will be disabled.")
            self.is_disabled = True
        else:
            self.is_disabled = False

    def _get_token_id(self, text):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Often these tokens are single tokens, but handle cases where they might be split
        return token_ids[0] if len(token_ids) == 1 else None

    def __call__(self, input_ids, scores):
        """
        This method is called at each generation step.
        `input_ids` are the tokens generated so far.
        `scores` are the logits for the next token.
        """
        if self.is_disabled:
            return scores

        # We only operate on the generated part of the input_ids
        # input_ids has shape (batch_size, sequence_length)
        generated_ids = input_ids[0][self.prompt_len:]
        
        # 1. Check if we are currently inside a <think> block
        in_think_block = self.think_token_id in generated_ids and self.end_think_token_id not in generated_ids

        # 2. Check if the block is too long and we should intervene
        if in_think_block and len(generated_ids) > self.max_think_length:
            # 3. Add a strong bias to the </think> token logit
            # scores has shape (batch_size, vocab_size)
            scores[0, self.end_think_token_id] += self.end_think_bias
            
        return scores

def build_openrouter_instruct(language: str, question: str):
    # This is a generic instruction for an API-based model
    # that understands system/user roles.
    # The <think> block instruction is preserved.
    system_prompt = f"""Below is an instruction that describes a task.
        Write a response that appropriately completes the request.
        Before generating the code, think carefully about the problem and write down a step-by-step plan in the <think> block.
        Then, complete the function, returning all completed function in a markdown codeblock. You are not allowed to modify the given code outside the completion.

        Complete the following {language} function:
        """
    user_prompt = f"""
        ### Given Code:
        ```{language.lower()}
        {question.strip()}
        ```

        ### Response:
        """
    return system_prompt, user_prompt.strip()

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
        """.strip()

def generate_one_openrouter(example, lang, client, model_name, max_new_tokens):
    system_prompt, user_prompt = build_openrouter_instruct(language_settings[lang]['full_name'], example['prompt'])
    
    print(f"\n[DEBUG][generate_one_openrouter] Task ID: {example.get('task_id', 'N/A')}")
    print(f"[DEBUG][generate_one_openrouter] System Prompt sent to API:\n{system_prompt}")
    print(f"[DEBUG][generate_one_openrouter] User Prompt sent to API:\n{user_prompt}")

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )
        output = completion.choices[0].message.content
        print(f"[DEBUG][generate_one_openrouter] Raw API output for Task ID {example.get('task_id', 'N/A')}:\n{output}")
        
    except Exception as e:
        print(f"Error calling OpenRouter API for task {example.get('task_id', 'N/A')}: {e}")
        output = f"<think>\nAPI Call Failed.\n</think>\n```{lang.lower()}\n# Error during generation\n```"

    example['output'] = output
    gen_example, success = extract_generation_code(example, lang_code=lang)
    print(f"[DEBUG][generate_one_openrouter] Extraction Code for Task ID {example.get('task_id', 'N/A')}:\n{gen_example.get('generation', 'N/A')}")
    return gen_example, success

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

    prompt_token_len = inputs.shape[-1]

    logits_processor = LogitsProcessorList([
        EndThinkBlockProcessor(
            tokenizer=tokenizer, 
            prompt_len=prompt_token_len,
            max_think_length=max_new_tokens - 1024,
            end_think_bias=5.0
        )
    ])

    with torch.inference_mode():
        with torch.autocast(device_type=model.device.type, dtype=torch.float16):
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[-1] + max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                pad_token_id=stop_id,
                eos_token_id=stop_id,
                logits_processor=logits_processor,
            )

    output = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"[DEBUG][generate_one] Raw model output for Task ID {example.get('task_id', 'N/A')}:\n{output}")
    example['output'] = output
    
    # Return both the example and the success status
    gen_example, success = extract_generation_code(example, lang_code=lang)
    print(f"[DEBUG][generate_one] Extraction Code for Task ID {example.get('task_id', 'N/A')}:\n{gen_example.get('generation', 'N/A')}")
    return gen_example, success

def save_results(args, result_data, failed_extractions=0):
    """
    Saves the evaluation results, model, and generation parameters to a JSON file.
    """
    results_dir = Path("result")
    results_dir.mkdir(parents=True, exist_ok=True) # Ensure the result directory exists

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model.replace("/", "_").replace("-", "_")
    output_filename = f"{model_name_safe}_{args.language}_{timestamp}.json"
    output_filepath = results_dir / output_filename

    save_data = {
        "timestamp": timestamp,
        "model": args.model,
        "language": args.language,
        "max_new_tokens": args.max_new_tokens,
        "evaluation_results": result_data,
        "failed_code_extractions": failed_extractions,
        "use_openrouter": args.use_openrouter,
        "use_vllm": args.use_vllm,
        "tokenizer_path": args.tokenizer_path
    }
    
    with open(output_filepath, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"[INFO] Evaluation results saved to {output_filepath}")

def generate_main(args):
    model_name_or_path = args.model
    lang = args.language
    saved_path = args.output_path
    temp_dir = args.temp_dir
    max_new_tokens = args.max_new_tokens
    presence_penalty = args.presence_penalty
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")

    print(f"[DEBUG][generate_main] lang: {lang}")
    print(f"[DEBUG][generate_main] max_new_tokens: {max_new_tokens}")
    print(f"[DEBUG][generate_main] Problem file: {problem_file}")

    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    print(f"Read {len(examples)} examples for evaluation over.")

    if args.use_openrouter:
        print("[INFO] Using OpenRouter API for generation.")
        api_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key must be provided via --openrouter_api_key or OPENROUTER_API_KEY environment variable.")
        
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print(f"[DEBUG][generate_main] OpenRouter model: {model_name_or_path}")

        generated_examples = []
        failed_extraction_count = 0
        for ex in tqdm(examples, desc='Generating with OpenRouter'):
            print(f"\n[DEBUG][generate_main] Processing Task ID: {ex.get('task_id', 'N/A')}")
            gen_example, extraction_successful = generate_one_openrouter(ex, lang, client, model_name_or_path, max_new_tokens)
            if not extraction_successful:
                failed_extraction_count += 1
            generated_examples.append(gen_example)

    else:
        print(f"[DEBUG][generate_main] model: {model_name_or_path}")
        
        if args.use_vllm:
            print("[INFO] Using vLLM for generation.")
            print("[WARN] The 'EndThinkBlockProcessor' is not used with vLLM, which may affect generation for very long thought processes.")
            
            num_gpus = torch.cuda.device_count()
            print(f"[INFO] Initializing vLLM with tensor_parallel_size={num_gpus}")
            llm = LLM(model=model_name_or_path, tensor_parallel_size=num_gpus)
            
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=max_new_tokens,
                stop=tokenizer.eos_token,
                presence_penalty=presence_penalty,
            )

            prompts = [
                build_deepseekcoder_instruction(language_settings[lang]['full_name'], ex['prompt'])
                for ex in examples
            ]
            
            print(f"Generating completions for {len(prompts)} prompts...")
            vllm_outputs = llm.generate(prompts, sampling_params)
            print("Generation complete.")
            
            vllm_outputs.sort(key=lambda o: int(o.request_id))

            generated_examples = []
            failed_extraction_count = 0
            for ex, vllm_output in tqdm(zip(examples, vllm_outputs), total=len(examples), desc="Processing vLLM outputs"):
                output = vllm_output.outputs[0].text
                print(f"[DEBUG][vLLM] Raw model output for Task ID {ex.get('task_id', 'N/A')}:\n{output}")
                
                ex['output'] = output
                gen_example, success = extract_generation_code(ex, lang_code=lang)
                if not success:
                    failed_extraction_count += 1
                generated_examples.append(gen_example)
        else:
            # Logic for local HuggingFace models
            if model_name_or_path.endswith('.pt'):
                print(f"Loading model from .pt file: {model_name_or_path}")
                if not args.tokenizer_path:
                    raise ValueError("When loading a .pt model file, --tokenizer_path must be provided.")
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
                print("load tokenizer {} from {} over.".format(tokenizer.__class__, args.tokenizer_path))
                model = torch.load(model_name_or_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                print("load tokenizer {} from {} over.".format(tokenizer.__class__, model_name_or_path))
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            
            model = torch.compile(model)
            model.eval()
            
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
    
    # Save the results
    save_results(args, result, failed_extraction_count)

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
    
    # Save the results
    save_results(args, result) # No failed_extractions count for evaluation_only

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name or path for local/HF, or model name on OpenRouter")
    parser.add_argument('--tokenizer_path', type=str, default=None, help="Path to tokenizer, needed if loading a .pt model file.")
    parser.add_argument('--output_path', type=str, help="output path of your generation")
    parser.add_argument('--language', type=str, help="langauge")
    parser.add_argument('--temp_dir', type=str, help="temp dir for evaluation", default="tmp")
    parser.add_argument('--max_new_tokens', type=int, help="max new tokens", default=4096)
    parser.add_argument('--evaluation_only', action='store_true', help="if only evaluate the output file")
    parser.add_argument('--presence_penalty', type=float, default=1.5, help="Presence penalty for sampling.")
    
    # OpenRouter specific arguments
    parser.add_argument('--use_openrouter', action='store_true', help="use OpenRouter API for generation")
    parser.add_argument('--openrouter_api_key', type=str, default=None, help="OpenRouter API key. Can also be set via OPENROUTER_API_KEY env var.")
    parser.add_argument('--use_vllm', action='store_true', help="Use vLLM for local generation")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.evaluation_only:
        evaluation_only(args)
    else:
        generate_main(args)
    pass
