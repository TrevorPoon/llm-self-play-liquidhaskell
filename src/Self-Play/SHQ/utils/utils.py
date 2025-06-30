import re
import traceback
import json
import subprocess
import torch


language_settings = {
    'python': {
        'full_name': 'Python',
        'indent': 4,
    },
    'cpp': {
        'full_name': 'cpp',
        'indent': 0,
        'main': "int main()",
    },
    'java': {
        'full_name': 'Java',
        'indent': 4,
        'main': "public static void main",
    },
    'cs': {
        'full_name': "csharp",
        'indent': 0,
        'main': "public static void Main",
    },
    'php': {
        'full_name': "PHP",
        'indent': 0,
    },
    'ts': {
        'full_name': "TypeScript",
        'indent': 0,
    },
    'js': {
        'full_name': "JavaScript",
        'indent': 0
    },
    'sh': {
        'full_name': "Bash",
        'indent': 0
    },
    'hs': {
        'full_name': 'Haskell',
        'indent': 0,
        'main': 'main :: IO ()'
    }
}

def get_function_arg_type(program_code: str):
    """
    Extracts the first argument type from a Haskell function's type signature.
    It handles simple signatures, signatures with typeclass constraints, and higher-order functions.
    e.g., `func :: Int -> Int` -> `Int`
    e.g., `func :: (Read a) => [a] -> [a]` -> `[a]`
    e.g., `func :: (Int -> Bool) -> [Int] -> [Int]` -> `(Int -> Bool)`
    e.g., `func :: (Int, String) -> Bool` -> `(Int, String)`
    """
    # Find the type signature part of the function definition
    match = re.search(r"^\s*[\w']+\s*::\s*(.*)", program_code, re.MULTILINE)
    if not match:
        return None
    
    signature = match.group(1).strip()

    # Remove typeclass constraints if they exist
    if '=>' in signature:
        signature = signature.split('=>', 1)[1].strip()

    paren_level = 0
    
    # Find the first '->' at the top level (paren_level 0)
    for i in range(len(signature) - 1):
        char = signature[i]
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
        elif signature[i:i+2] == '->' and paren_level == 0:
            # We found the top-level arrow. The part before it is the first argument type.
            return signature[:i].strip()

    return None

def get_function_name(program_code: str):
    """Extracts function name from a Haskell code snippet."""
    match = re.search(r"^([\w']+)\s*::", program_code, re.MULTILINE)
    if match:
        return match.group(1)
    return None

def extract_generation_code(output: str) -> str:
    """Extracts Haskell code from a markdown block."""
    try:
        # Use a more robust regex to find the code block
        code_block_match = re.search(r"```haskell\n(.*?)\n```", output, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            return code_block_match.group(1).strip()
    except Exception as e:
        print(f"Failed to extract code block with error: {e}")
    return None

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    print(f"[DEBUG][cleanup_code] Before Cleanup Code:\n---\n{code}\n---")

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)
    
    print(f"[DEBUG][cleanup_code] After Cleanup Code:\n---\n{code}\n---")

    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]

def print_nvidia_smi(label=""):
    """Prints the current GPU memory usage using nvidia-smi"""
    nvidia_smi_output = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
    print(f"Current GPU Memory Usage ({label}):\n{nvidia_smi_output}")
    
def print_gpu_memory_usage(label=""):
    """Prints detailed GPU memory usage for each GPU."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    print(f"\n--- GPU Memory Usage ({label}) ---")
    for i in range(torch.cuda.device_count()):
        allocated_mem = torch.cuda.memory_allocated(i) / 1024**3
        reserved_mem = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated_mem = torch.cuda.max_memory_allocated(i) / 1024**3
        
        print(f"--- GPU {i}: {torch.cuda.get_device_name(i)} ---")
        print(f"  - Allocated memory: {allocated_mem:.2f} GB")
        print(f"  - Reserved memory (from PyTorch): {reserved_mem:.2f} GB")
        print(f"  - Peak allocated memory: {max_allocated_mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats(i) # Reset peak stats after printing
    print("-------------------------------------\n")