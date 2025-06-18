#!/usr/bin/env python3
"""
Script to load multi-line JSONL file of Haskell human-eval problems.
Reads the file at the given path, accumulates complete JSON objects
across multiple physical lines, and parses them into Python dicts.
"""
import json
import sys


def load_jsonl_multiline(path):
    """
    Read a JSONL file where each logical JSON object may span multiple lines.
    Returns a list of parsed dicts.
    """
    examples = []
    buffer = []
    depth = 0

    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            # Update brace depth
            depth += line.count('{') - line.count('}')
            buffer.append(line)

            # When depth is zero, we have a complete JSON object
            if depth == 0 and buffer:
                text = ''.join(buffer).strip()
                if text:
                    try:
                        examples.append(json.loads(text))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON at lines {lineno - len(buffer) + 1}-{lineno}: {e}", file=sys.stderr)
                        print("Offending text:\n", text, file=sys.stderr)
                        sys.exit(1)
                buffer = []

    return examples


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/humaneval-hs.jsonl")
        sys.exit(1)

    path = sys.argv[1]
    examples = load_jsonl_multiline(path)
    print(f"Loaded {len(examples)} examples from {path}.")

    # Example: print the first problem_id and prompt
    if examples:
        first = examples[0]
        print("First task_id:", first.get('task_id'))
        print("Prompt snippet:\n", first.get('prompt', '')[:200], '...')


if __name__ == '__main__':
    main()
