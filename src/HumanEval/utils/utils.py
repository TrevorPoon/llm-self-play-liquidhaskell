import re
import traceback
import json

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

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    if lang.lower() == 'haskell':
        # For Haskell, the function signature is often the last line.
        # e.g., `my_function :: [Int] -> Int`
        signature_line = func_lines[-1]
        func_name = signature_line.split('::')[0].strip()
        func_prefix = "\n".join(func_lines[:-1])
        return func_name, func_prefix

    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(example: str, lang_code: str, verbose: bool=True):
    task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    print(f"[DEBUG][extract_generation_code] Processing task: {task_id}, language: {lang_code}")
    print(f"[DEBUG][extract_generation_code] RAW MODEL OUTPUT for {task_id}:\n---\n{output}\n---")
    
    question = example["prompt"].strip()
    setting = language_settings[lang_code]
    lang = setting['full_name']
    indent = setting['indent']

    func_name, func_prefix = get_function_name(question, lang)
    print(f"[DEBUG][extract_generation_code] Extracted function name '{func_name}' and prefix:\n---\n{func_prefix}\n---")

    try:
        print(f"[DEBUG][extract_generation_code] Searching for code block using regex...")
        code_block_matches = re.findall(f'```{lang.lower()}?\\s*\n(.*?)\n?```', output, re.DOTALL | re.IGNORECASE)
        if not code_block_matches:
            print(f"[DEBUG][extract_generation_code] Initial regex failed. Trying fallback regex...")
            code_block_matches = re.findall(f'```\\s*\n(.*?)\n?```', output, re.DOTALL)

        if code_block_matches:
            print(f"[DEBUG][extract_generation_code] Found {len(code_block_matches)} code blocks. Using the last one.")
            code_block: str = code_block_matches[-1].strip()
            print(f"[DEBUG][extract_generation_code] EXTRACTED CODE BLOCK (from markdown) for {task_id}:\n---\n{code_block}\n---")
        else:
            print(f"[DEBUG][extract_generation_code] No markdown code block found. Trying fallback extraction methods.")
            if lang_code.lower() == 'hs':
                signature = f"{func_name} ::"
                if signature in output:
                    last_occurrence_index = output.rfind(signature)
                    code_block = output[last_occurrence_index:]
                    print(f"[DEBUG][extract_generation_code] EXTRACTED CODE BLOCK (Haskell fallback) for {task_id}:\n---\n{code_block}\n---")
                else:
                    raise IndexError("No code block found and no Haskell function signature found.")
            else:
                 raise IndexError("No code block found in model output")

        if lang_code.lower() == 'hs' and setting.get('main') and setting['main'] in code_block:
            print(f"[DEBUG][extract_generation_code] Haskell detected. Checking for main function '{setting['main']}'...")
            main_start = code_block.find(setting['main'])
            code_block = code_block[:main_start].strip()
            print(f"[DEBUG][extract_generation_code] CODE BLOCK after main removal for {task_id}:\n---\n{code_block}\n---")
        elif setting.get('main', None) and setting['main'] in code_block:
            print(f"[DEBUG][extract_generation_code] Language {lang_code}. Checking for main function '{setting['main']}'...")
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
            print(f"[DEBUG][extract_generation_code] CODE BLOCK after main removal for {task_id}:\n---\n{code_block}\n---")

        if lang_code.lower() == 'hs':
            body = code_block
        else:
            try:
                print(f"[DEBUG][extract_generation_code] Finding start of function body for '{func_name}'...")
                start = code_block.lower().index(func_name.lower())
                print(f"[DEBUG][extract_generation_code] Function name found at index {start}.")
                indent = 0
                while start - indent >= 0 and code_block[start - indent-1] == ' ':
                    indent += 1
                print(f"[DEBUG][extract_generation_code] Calculated indent: {indent}")
                
                try:
                    print(f"[DEBUG][extract_generation_code] Finding end of function body...")
                    end = code_block.rindex('\n' + ' '*indent + '}')
                    print(f"[DEBUG][extract_generation_code] Found end of function body at index {end}.")
                except:
                    end = len(code_block)
                    print(f"[DEBUG][extract_generation_code] Could not find '}}' with indent, setting end to length of code_block ({end}).")
            except:
                start = 0
                print(f"[DEBUG][extract_generation_code] Could not find function name. Setting start to 0.")
                try:
                    end = code_block.rindex('\n' + ' '*indent + '}')
                    print(f"[DEBUG][extract_generation_code] Found end of function body at index {end}.")
                except:
                    end = len(code_block)
                    print(f"[DEBUG][extract_generation_code] Could not find '}}' with indent, setting end to length of code_block ({end}).")

            body = code_block[start:end]
            print(f"[DEBUG][extract_generation_code] EXTRACTED FUNCTION BODY for {task_id}:\n---\n{body}\n---")

            if lang_code.lower() in ['php', 'ts', 'js']:
                print(f"[DEBUG][extract_generation_code] Appending '}}' for language {lang_code}.")
                body += '\n' + ' '*indent + '}'
    
        generation = func_prefix + '\n' + body + '\n'
        example['generation'] = generation

        print(f"[DEBUG][extract_generation_code] FINAL GENERATED CODE for {task_id}:\n---\n{generation}\n---")

        extraction_successful = True

    except Exception as ex:
        print(f"[DEBUG][extract_generation_code] Failed to extract code block with error `{ex}`:\n>>> Task: {task_id}\n>>> Output:\n{output}")
        print(f"[DEBUG][extract_generation_code] Traceback:\n{traceback.format_exc()}")
        example['generation'] = example['prompt'] + '\n' + output
        extraction_successful = False
    
    return example, extraction_successful

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

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    elif language_type.lower() == "haskell":
        stop_words = ["\nmain :: IO ()", "\ncheck ::"]
        code = _truncate_code_at_stopwords(code, stop_words)
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

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