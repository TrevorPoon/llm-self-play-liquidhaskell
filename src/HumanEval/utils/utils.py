import re

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
    'go': {
        'full_name': "Go",
        'indent': 0, # Typically uses tabs, but for extraction, 0 is fine as fallback.
    }
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(example: str, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    output = example.get('output', example.get("gpt_completion"))
    question = example["prompt"].strip()
    setting = language_settings[lang_code]
    lang = setting['full_name']
    extraction_successful = False

    try:
        # First, find the end of the </think> block
        think_end_tag = '</think>'
        think_end_index = output.find(think_end_tag)
        if think_end_index == -1:
            # If no </think> tag, try to extract code from the beginning as a fallback
            # This maintains compatibility or handles cases where the model might not produce the <think> block
            print(f"Warning: No </think> tag found in output for task {task_id}. Attempting direct code extraction.")
            code_start_offset = 0
        else:
            code_start_offset = think_end_index + len(think_end_tag)

        # Search for the code block after the </think> tag
        # Use re.IGNORECASE for the language marker to be more robust
        # Ensure we're looking for ```lang
        code_block_regex = f'```{re.escape(lang.lower())}\n(.*?)\n```'
        found_blocks = re.findall(code_block_regex, output[code_start_offset:], re.DOTALL | re.IGNORECASE)
        
        if not found_blocks:
            # Fallback: try to find ```lang ... ``` (without newline after lang)
            code_block_regex_no_newline = f'```{re.escape(lang.lower())}(.*?)```'
            found_blocks = re.findall(code_block_regex_no_newline, output[code_start_offset:], re.DOTALL | re.IGNORECASE)
            if found_blocks:
                 # If found without newline, extract the content, ensuring it's stripped if it started with a newline
                 raw_block = found_blocks[-1]
                 found_blocks = [raw_block.lstrip('\n')] 

        if not found_blocks:
            # Fallback: if the model just gives raw code after </think> without markdown
            # This is a more aggressive fallback and might need careful testing
            potential_code_after_think = output[code_start_offset:].strip()
            if potential_code_after_think and not potential_code_after_think.startswith("```") :
                print(f"Warning: No markdown code block found after </think> for task {task_id}. Assuming raw code.")
                found_blocks = [potential_code_after_think]
            else:
                 raise ValueError(f"No code blocks found for language {lang.lower()} after {think_end_tag if think_end_index != -1 else 'start of output'}")
        
        code_block: str = found_blocks[-1].strip() # Strip any leading/trailing whitespace from the extracted block
        
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))
        
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
        
        func_name, func_prefix = get_function_name(question, lang)

        try:
            start = code_block.lower().index(func_name.lower())
            current_func_indent_level = 0
            temp_start = start
            while temp_start > 0 and code_block[temp_start-1] == ' ':
                current_func_indent_level += 1
                temp_start -=1
            
            line_start_for_func_name = code_block.rfind('\n', 0, start) + 1
            calculated_indent_val = start - line_start_for_func_name

            try:
                original_calculated_indent = 0
                temp_idx = start
                while temp_idx > 0 and code_block[temp_idx-1] == ' ':
                    original_calculated_indent +=1
                    temp_idx -=1
                
                end = code_block.rindex('\n' + ' '*original_calculated_indent + '}')
            except ValueError:
                end = len(code_block)

        except ValueError:
            start = 0
            try:
                end = code_block.rindex('\n' + ' '*setting['indent'] + '}')
            except ValueError:
                end = len(code_block)

        body = code_block[start:end]

        if lang_code.lower() in ['php', 'ts', 'js']:
            final_indent_for_closing_brace = setting['indent']
            if 'original_calculated_indent' in locals():
                 final_indent_for_closing_brace = original_calculated_indent
            body += '\n' + ' '*final_indent_for_closing_brace + '}'
    
        generation = func_prefix + '\n' + body + '\n'
        example['generation'] = generation
        extraction_successful = True

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        example['generation'] = example['prompt'] + '\n' + output
    
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