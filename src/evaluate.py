# ===== File: src/evaluate.py =====

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils.logger import setup_logger
from inference import run_inference  # Assumed to return the generated output text


def load_humaneval_dataset():
    """
    Loads the OpenAI HumanEval dataset using Hugging Face's datasets library.
    Returns the test split containing 164 programming problems.
    """
    dataset = load_dataset("openai_humaneval", split="test")
    return dataset


def run_evaluation(model, tokenizer, device, logger):
    """
    Evaluates the model on the HumanEval tasks.
    
    For each task:
    - Runs inference using the provided prompt.
    - Executes the generated candidate code.
    - Sets the candidate function (using the entry_point field) into the local environment.
    - Executes the test code provided in the dataset to determine correctness.
    
    Returns a list of evaluation results for each task.
    """

    # Load HumanEval dataset
    dataset = load_humaneval_dataset()
    logger.info(f"Loaded HumanEval dataset with {len(dataset)} tasks.")

    results = []

    for task in dataset:
        task_id = task["task_id"]
        prompt = task["prompt"]
        test_code = task["test"]
        entry_point = task["entry_point"]

        logger.info(f"Evaluating Task {task_id}")
        logger.info(f"Prompt:\n{prompt}")

        # Run inference (run_inference should return output_text)
        output_text = run_inference(model, tokenizer, prompt, device, logger)
        logger.info(f"Generated Code for Task {task_id}:\n{output_text}")

        # Prepare a local environment for executing the generated code and test code
        local_env = {}
        passed = False
        error_message = None
        try:
            # Execute the generated code to define the candidate function.
            exec(output_text, local_env)
            if entry_point not in local_env:
                raise ValueError(f"Candidate function '{entry_point}' not found in generated code.")

            # Bind the candidate function to a known name 'candidate' for the test.
            candidate = local_env[entry_point]
            local_env["candidate"] = candidate

            # Execute the test code; it should define a test function (e.g., 'check')
            # and then invoke it with the candidate function.
            exec(test_code, local_env)
            if "check" not in local_env:
                raise ValueError("Test function 'check' not defined in test code.")
            # Run the test; any assertion error will be caught below.
            local_env["check"](candidate)
            passed = True
        except Exception as e:
            error_message = str(e)
            logger.error(f"Task {task_id} failed with error: {error_message}")

        results.append({
            "task_id": task_id,
            "pass": passed,
            "error": error_message,
        })
        logger.info(f"Task {task_id} - Passed: {passed}")

        # Summarize evaluation results
    total = len(results)
    passed_count = sum(1 for res in results if res.get("pass"))
    logger.info("----- Evaluation Summary -----")
    logger.info(f"Total Tasks: {total}")
    logger.info(f"Tasks Passed: {passed_count}")
    logger.info(f"Tasks Failed: {total - passed_count}")

    # Optionally, print detailed results for each task
    for res in results:
        logger.info(f"Task {res['task_id']} - Passed: {res['pass']}, Error: {res.get('error', 'None')}")

    return results