import argparse
import logging
import json
import os
import datetime
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import load_dataset
import re
import textwrap
import tqdm

SYSTEM_PROMPT = textwrap.dedent("""You are an expert in programming. 
    Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack. We treat the task as binary classification (true/false), where true stands for insecure code and false stands for secure code.
    Your response must be either "true" or "false" at the very end of your output. 
    Please output your final answer in the format 
    ```
    true
    ```
    or 
    ```
    false
    ```
    """) 

USER_PROMPT = textwrap.dedent("""Is the code insecure?
    ```c
    {program}
    ```
    <think>
    """)


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic equivalence of Haskell programs using vLLM.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Sampling top-p.")
    parser.add_argument("--top_k", type=int, default=20, help="Sampling top-k.")
    parser.add_argument("--presence_penalty", type=float, default=1.5, help="Sampling presence penalty.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="The model to use for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=32768, help="The maximum number of new tokens to generate.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the LoRA adapter.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument("--num_iterations", type=int, default=1, help="The number of samples to evaluate.")
    args = parser.parse_args()


    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    dataset = load_dataset("google/code_x_glue_cc_defect_detection", split="test")

    logging.info(f"Initializing LLM with model: {args.model}, LoRA path: {args.adapter_path}, GPU memory utilization: {args.gpu_memory_utilization}")
    # Initialize vLLM
    llm = LLM(model=args.model, 
              enable_lora=True, 
              gpu_memory_utilization=args.gpu_memory_utilization)
    
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_new_tokens
    )

    for i in range(args.num_iterations):

        logging.info(f"Iteration {i+1} of {args.num_iterations}")


        all_predictions = []
        all_truth_labels = []
        total_samples = 0
        unparsed_samples = 0

        for item in tqdm.tqdm(dataset, desc=f"Processing dataset"):

            total_samples += 1

            program = item['func']
            truth_label = item['target']

            user_prompt = USER_PROMPT.format(program=program)

            # Combine prompts for inference
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"


            try:
                # Perform inference
                output = llm.generate(
                    [full_prompt], 
                    sampling_params, 
                    lora_request=LoRARequest("adapter", 1, args.adapter_path) if args.adapter_path else None
                )
                model_response = output[0].outputs[0].text.strip().lower()

                # Remove markdown formatting and extract the last boolean-like token
                match = re.search(r"(true|false|yes|no|1|0)\s*```?", model_response, re.IGNORECASE)
                if not match:
                    match = re.search(r"```(?:\s*\n)?(true|false|yes|no|1|0)\s*```?", model_response, re.IGNORECASE)
                if not match:
                    match = re.search(r"<answer>(true|false|1|0)</answer>", model_response, re.IGNORECASE)
                if not match:
                    match = re.search(r"(true|false|yes|no|1|0)\b", model_response)

                if match:
                    prediction_text = match.group(1).strip().lower()
                    prediction = prediction_text in ["true", "yes", "1"]
                else:
                    unparsed_samples += 1
                    logging.info(f"Unparsed sample: {model_response}")
                    continue

                all_predictions.append(prediction)
                all_truth_labels.append(truth_label)
                
            except Exception as e:
                logging.exception(f"Error processing sample {item}: {e}")
                unparsed_samples += 1
                continue


        # Compute evaluation metrics
        results = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "setup": {
                "model": args.model,
                "adapter_path": args.adapter_path,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "sampling_parameters": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "presence_penalty": args.presence_penalty,
                    "max_new_tokens": args.max_new_tokens
                }
            },
            "iteration": i,
            "datasets": "codexglue",
            "evaluation_metrics": {},
            "predictions_summary": {
                "total_samples": total_samples,
                "parsed_samples": len(all_predictions),
                "unparsed_samples": unparsed_samples
            }
        }

        if len(all_predictions) > 0:
            accuracy = accuracy_score(all_truth_labels, all_predictions)
            precision = precision_score(all_truth_labels, all_predictions)
            recall = recall_score(all_truth_labels, all_predictions)
            f1 = f1_score(all_truth_labels, all_predictions)
            conf_matrix = confusion_matrix(all_truth_labels, all_predictions).tolist() # Convert to list for JSON serialization

            results["evaluation_metrics"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix
            }

            logging.info("\n--- Evaluation Metrics ---")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")
        else:
            logging.info("No predictions were made to evaluate.")

        # Save results to JSON file
        safe_model = args.model.replace(os.sep, "_").replace("/", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{safe_model}_evaluation_results_{timestamp}_iters{args.num_iterations}.json"
        output_path = os.path.join(results_dir, fname)
        tmp_path = output_path + ".tmp"

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, output_path)
            logging.info(f"Results saved to {output_path}")
        except Exception as e:
            logging.exception(f"Failed to save results to {output_path}: {e}")

if __name__ == "__main__":
    main()

