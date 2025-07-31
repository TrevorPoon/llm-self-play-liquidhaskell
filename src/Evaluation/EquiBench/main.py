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

SYSTEM_PROMPT = textwrap.dedent("""You are an expert in programming. Your task is to determine if two given Haskell programs are semantically equivalent. 
    Respond with "true" if they are equivalent, and "false" otherwise. 
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

USER_PROMPT = textwrap.dedent("""Are the following two programs semantically equivalent?
    Program 1:
    ```
    {program_1}
    ```
    Program 2:
    ```
    {program_2}
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
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="The maximum number of new tokens to generate.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the LoRA adapter.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="The fraction of GPU memory to be used for the vLLM KV cache.")
    parser.add_argument("--num_iterations", type=int, default=8, help="The number of samples to evaluate.")
    args = parser.parse_args()


    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    configs = ['DCE', 'OJ_A', 'OJ_V', 'OJ_VA', 'STOKE', 'TVM']
    all_datasets = {cfg: load_dataset("anjiangwei/EquiBench-Datasets", cfg, split="train") for cfg in configs}

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



    all_predictions = []
    all_truth_labels = []

    for i in range(args.num_iterations):

        logging.info(f"Iteration {i+1} of {args.num_iterations}")

        for cfg in configs:
            dataset = all_datasets[cfg]
            logging.info(f"Processing dataset: {cfg}")

            for item in tqdm.tqdm(dataset, desc=f"Processing {cfg} set"):

                program_1 = item['program_1_code']
                program_2 = item['program_2_code']
                truth_label = item['truth_label']

                user_prompt = USER_PROMPT.format(program_1=program_1, program_2=program_2)

                # Combine prompts for inference
                full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

                # Perform inference
                output = llm.generate(
                    [full_prompt], 
                    sampling_params, 
                    lora_request=LoRARequest("adapter", 1, args.adapter_path) if args.adapter_path else None
                )
                model_response = output[0].outputs[0].text.strip().lower()

                # Parse prediction
                prediction = None
                # Look for the designated answer format first
                match = re.search(r"```\s*(true|yes)\s*```", model_response, re.IGNORECASE)
                if match:
                    prediction = True
                else:
                    match = re.search(r"```\s*(false|no)\s*```", model_response, re.IGNORECASE)
                    if match:
                        prediction = False
                
                if prediction is None:
                    # logging.warning(f"Could not parse model response: {model_response}. Skipping this sample.")
                    continue

                all_predictions.append(prediction)
                all_truth_labels.append(truth_label)

                # Print example queries and responses (Optional Bonus)
                # if len(all_predictions) <= 5: # Print first 5 examples
                #     logging.info(f"\n--- Example {len(all_predictions)} ---")
                #     logging.info(f"User Prompt:\n{user_prompt}")
                #     logging.info(f"Model Response: {model_response}")
                #     logging.info(f"Parsed Prediction: {prediction}")
                #     logging.info(f"Truth Label: {truth_label}")

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
                "datasets": cfg,
                "evaluation_metrics": {},
                "predictions_summary": {
                    "total_samples": len(all_predictions),
                    "parsed_samples": len(all_predictions),
                    "unparsed_samples": 0 # This needs to be tracked if implemented properly
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(results_dir, f"{args.model}_evaluation_results_{timestamp}_iterations{args.num_iterations}_cfg{cfg}.json")
        with open(output_filename, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {output_filename}")

if __name__ == "__main__":
    main()

