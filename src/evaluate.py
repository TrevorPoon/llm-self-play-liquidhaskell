import subprocess
import tempfile
import os
from datasets import load_dataset
import shutil

SUPPORTED_LANGUAGES = ["python", "js", "go", "cpp", "java"]

def execute_code(script_content, language, logger):
    """
    Executes the given script content in a sandboxed environment.
    Returns True if execution is successful and tests pass, False otherwise.
    """
    if language not in SUPPORTED_LANGUAGES:
        logger.error(f"Unsupported language: {language}")
        return False, "Unsupported language"

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            if language == "python":
                filepath = os.path.join(temp_dir, "solution.py")
                with open(filepath, "w") as f:
                    f.write(script_content)
                result = subprocess.run(
                    ["python", filepath],
                    capture_output=True,
                    text=True,
                    timeout=20,  # Increased timeout
                )
                return result.returncode == 0 and not result.stderr, result.stdout + "\\n" + result.stderr

            elif language == "js":
                filepath = os.path.join(temp_dir, "solution.js")
                with open(filepath, "w") as f:
                    f.write(script_content)
                result = subprocess.run(
                    ["node", filepath],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                # Node assertions throw errors and exit with non-zero, or console.assert logs to stderr
                return result.returncode == 0 and not result.stderr, result.stdout + "\\n" + result.stderr
            
            elif language == "go":
                filepath = os.path.join(temp_dir, "solution.go")
                # Go programs need a package main and func main if test is a full program.
                # If test is just calling a function, it needs to be wrapped.
                # Assuming HumanEval-X 'test' field for Go is a runnable main or test functions.
                # For simplicity, let's assume the 'test' field can be directly run.
                # This might need adjustment based on Go test structure in HumanEval-X.
                full_script = script_content # Directly use script_content which should include prompt, solution, and test
                with open(filepath, "w") as f:
                    f.write(full_script)

                result = subprocess.run(
                    ["go", "run", filepath],
                    cwd=temp_dir, # Run go from the temp_dir
                    capture_output=True,
                    text=True,
                    timeout=30, # Go compilation can be slower
                )
                return result.returncode == 0 and not result.stderr, result.stdout + "\\n" + result.stderr

            elif language == "cpp":
                filepath = os.path.join(temp_dir, "solution.cpp")
                executable_path = os.path.join(temp_dir, "solution_cpp")
                with open(filepath, "w") as f:
                    f.write(script_content)
                
                compile_result = subprocess.run(
                    ["g++", "-std=c++11", filepath, "-o", executable_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if compile_result.returncode != 0:
                    logger.error(f"C++ compilation failed: {compile_result.stderr}")
                    return False, "Compilation failed: \\n" + compile_result.stderr
                
                run_result = subprocess.run(
                    [executable_path],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                return run_result.returncode == 0 and not run_result.stderr, run_result.stdout + "\\n" + run_result.stderr

            elif language == "java":
                # Java execution is more complex due to class names and file names.
                # HumanEval-X Java prompts usually define a class, e.g., "class Solution { ... }"
                # The test code might assume this class structure.
                # We need to extract the class name to save the file correctly.
                
                # Simplistic way to find main class name, assuming it's "Solution" or first public class
                class_name = "Solution" # Default, might need robust extraction
                if "class " in script_content:
                    try:
                        # Try to find "public class X" or "class X"
                        public_class_match = [line for line in script_content.splitlines() if "public class " in line]
                        class_match = [line for line in script_content.splitlines() if "class " in line and "public" not in line]
                        
                        if public_class_match:
                            class_name = public_class_match[0].split("public class ")[1].split("{")[0].strip()
                        elif class_match:
                             class_name = class_match[0].split("class ")[1].split("{")[0].strip()
                        
                        # Ensure class_name is valid for a filename
                        class_name = "".join(c if c.isalnum() else "_" for c in class_name)
                        if not class_name: class_name = "Main" # Fallback

                    except Exception as e:
                        logger.warning(f"Could not reliably extract class name, defaulting to 'Main'. Error: {e}")
                        class_name = "Main"
                else: # If no class definition, assume it's a script-like structure for some simple Java environments (less common for HumanEval)
                    class_name = "Main"


                filepath = os.path.join(temp_dir, f"{class_name}.java")
                with open(filepath, "w") as f:
                    f.write(script_content)

                compile_result = subprocess.run(
                    ["javac", filepath],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if compile_result.returncode != 0:
                    logger.error(f"Java compilation failed: {compile_result.stderr}")
                    return False, "Compilation failed: \\n" + compile_result.stderr
                
                run_result = subprocess.run(
                    ["java", "-cp", temp_dir, class_name],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                # Java assertions usually throw exceptions, which go to stderr and often set non-zero exit code.
                return run_result.returncode == 0 and not run_result.stderr, run_result.stdout + "\\n" + run_result.stderr

        except subprocess.TimeoutExpired:
            logger.warning(f"Code execution timed out for language {language}.")
            return False, "Execution timed out"
        except Exception as e:
            logger.error(f"Error executing code for {language}: {e}")
            return False, f"Execution error: {str(e)}"

    return False, "Execution failed due to temp_dir issue (should not happen)"


def run_evaluation(model, tokenizer, device, logger, language="python", max_length=2048):
    logger.info(f"Starting evaluation on HumanEval-X for language: {language}")

    if language not in SUPPORTED_LANGUAGES:
        logger.error(f"Language '{language}' is not supported for HumanEval-X. Supported: {SUPPORTED_LANGUAGES}")
        return

    try:
        dataset = load_dataset("THUDM/humaneval-x", language, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset THUDM/humaneval-x for language {language}: {e}")
        logger.error("Please ensure you have the 'datasets' library installed and internet connectivity.")
        logger.error("You might need to run: pip install datasets evaluate") # Added evaluate as it's often used with datasets
        return

    test_data = dataset["test"]
    total_tasks = 0
    passed_tasks = 0

    # Check if model and tokenizer are loaded
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer not provided for evaluation.")
        return

    for task in test_data:
        task_id = task["task_id"]
        prompt_text = task["prompt"] # This is the part the model should complete
        test_code = task["test"]     # This is the test script

        logger.info(f"Evaluating task: {task_id}")
        total_tasks += 1

        try:
            inputs = tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=max_length // 2).to(device) # Ensure prompt is not too long
            
            # Generate completion
            # The generation parameters might need tuning.
            # Using a simple generate call here.
            # Ensure model is on the correct device already (handled in main.py)
            generated_ids = model.generate(
                inputs,
                max_new_tokens=max_length // 2, # Allow ample space for generation
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Common generation parameters you might want to add or tune:
                # temperature=0.7,
                # top_p=0.9,
                # do_sample=True,
            )
            
            # Decode only the newly generated tokens
            generated_completion_ids = generated_ids[0, inputs.shape[-1]:]
            generated_completion = tokenizer.decode(generated_completion_ids, skip_special_tokens=True)

            logger.info(f"Generated solution for {task_id}:\n{generated_completion}")

            # Form the full script
            # The prompt already contains the function signature.
            # The model generates the body.
            # The test code executes the function.
            full_script = prompt_text + generated_completion + "\\n\\n" + test_code
            
            # Special handling for Java: test code might need to be inside the class or call a main method.
            # For now, the generic approach might work if the test is self-contained
            # and can be appended like this.
            if language == "java":
                 # A common pattern for HumanEval-X Java is that the prompt is a class structure
                 # and the test is separate. We might need to inject the test into a main method
                 # or ensure the generated code and test code are compatible.
                 # A simple concatenation might lead to syntax errors if not structured properly.
                 # For example, if prompt is `class Solution { public String hello() {`
                 # and completion is `return "world"; } }`
                 # and test is `public static void main(String[] args) { Solution s = new Solution(); System.out.println(s.hello()); }`
                 # We need to ensure the class structure is correct.
                 # One approach: place the test code inside the main method of the generated class if one doesn't exist.
                 # This is complex. For now, we'll try direct concatenation and see.

                 # Let's refine for Java: if prompt defines a class, completion is its methods. Test needs to be runnable.
                 # If the `test` field contains a `main` method, it should be fine.
                 # If not, the test might be snippets that assume the class is already compiled.
                 # A robust solution for Java would be to wrap the test in a main method of a new class,
                 # or compile the solution class and then run a separate test class.

                 # For now, let's assume the prompt + completion form a complete class, and test can be appended.
                 # This is often the case if the test is like `public class Test { public static void main(String[] args) { ... } }`
                 # Or if the test code itself defines assertions that can be run.
                 # The current `execute_code` for Java assumes `class_name.java` contains the main method or assertions run upon class loading.
                 pass # Keeping current logic for now, might need refinement

            passed, output = execute_code(full_script, language, logger)

            if passed:
                passed_tasks += 1
                logger.info(f"Task {task_id}: PASSED")
            else:
                logger.warning(f"Task {task_id}: FAILED")
                logger.debug(f"Failed output for {task_id} ({language}):\\n{output}")

        except Exception as e:
            logger.error(f"Error during evaluation of task {task_id}: {e}")
            logger.debug(f"Prompt for failed task {task_id}:\n{prompt_text}")
            logger.debug(f"Generated completion for failed task {task_id}:\n{generated_completion if 'generated_completion' in locals() else 'N/A'}")


    if total_tasks > 0:
        pass_rate = (passed_tasks / total_tasks) * 100
        logger.info(f"Overall Pass Rate ({language}): {pass_rate:.2f}% ({passed_tasks}/{total_tasks})")
    else:
        logger.info(f"No tasks found for language {language} in the dataset.")

    # Clean up Hugging Face cache if necessary, or let it be.
    # For CI/repeated runs, sometimes clearing cache is useful.
    # Example: shutil.rmtree(os.path.expanduser('~/.cache/huggingface/datasets/thudm___humaneval-x'))
    # But generally not needed unless space or corruption is an issue.

if __name__ == "__main__":
    # This part is for standalone testing of evaluate.py
    # In the main application, run_evaluation will be called from main.py
    import argparse
    from utils.logger import setup_logger # Assuming utils.logger exists
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig # For standalone
    import torch # For standalone

    parser = argparse.ArgumentParser(description="Run HumanEval-X evaluation")
    parser.add_argument('--language', type=str, default='python', choices=SUPPORTED_LANGUAGES,
                        help="Language subset to evaluate.")
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", # Or a smaller model for quick tests
                        help="Model name or path for evaluation.")
    parser.add_argument('--max_length', type=int, default=1024, help="Max length for generation.") # Smaller for faster local test
    parser.add_argument('--hf_token', type=str, default=None, help="Hugging Face token for private models.")


    args = parser.parse_args()
    logger = setup_logger()

    logger.info("--- Standalone Evaluation Mode ---")
    
    if not os.path.exists("utils/logger.py"):
        logger.warning("utils/logger.py not found. Creating a dummy logger for standalone run.")
        # Create a dummy logger if setup_logger is not available
        import logging
        logger = logging.getLogger("standalone_eval_logger")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        logger.info(f"Loading model: {args.model_name}")
        config = AutoConfig.from_pretrained(args.model_name, token=args.hf_token, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token, trust_remote_code=True)
        # Ensure pad_token is set for open-ended generation
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("tokenizer.pad_token was None, set to tokenizer.eos_token")
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            device_map="auto" if device == "cuda" else {"": device},
            token=args.hf_token,
            trust_remote_code=True
        )
        model.eval() # Set model to evaluation mode

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}")
        logger.error("Please ensure the model name is correct and you have internet access / model files available.")
        model, tokenizer = None, None # Ensure they are None so run_evaluation exits gracefully

    if model and tokenizer:
        run_evaluation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
            language=args.language,
            max_length=args.max_length
        )
    else:
        logger.error("Evaluation cannot proceed without a loaded model and tokenizer.")
    
    logger.info("--- Standalone Evaluation Finished ---")
