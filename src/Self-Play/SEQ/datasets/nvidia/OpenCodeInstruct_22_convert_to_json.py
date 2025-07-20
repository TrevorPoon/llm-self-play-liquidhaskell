from typing import *

def convert_to_json(data_structure):
    """
    Converts any data structure (list, dict, tuple, etc.) into a JSON-formatted string.
    
    :param data_structure: A valid data structure that can be converted to JSON.
    :return: String representation of the data structure in JSON format.
    """
    try:
        return json.dumps(data_structure, indent=4)
    except (TypeError, ValueError) as e:
        return f"Error converting to JSON: {str(e)}"

### Unit tests below ###
def check(candidate):
    assert candidate({"key": "value"}) == '{\n    "key": "value"\n}'
    assert candidate([1, 2, 3]) == '[\n    1,\n    2,\n    3\n]'
    assert candidate((1, 2, 3)) == '[\n    1,\n    2,\n    3\n]'
    assert candidate({"nested": {"key": "value"}}) == '{\n    "nested": {\n        "key": "value"\n    }\n}'
    assert candidate([{"key": "value"}, {"another_key": "another_value"}]) == '[\n    {\n        "key": "value"\n    },\n    {\n        "another_key": "another_value"\n    }\n]'
    assert candidate(({"key": "value"}, {"another_key": "another_value"})) == '[\n    {\n        "key": "value"\n    },\n    {\n        "another_key": "another_value"\n    }\n]'
    assert candidate({"list": [1, 2, 3], "tuple": (4, 5, 6)}) == '{\n    "list": [\n        1,\n        2,\n        3\n    ],\n    "tuple": [\n        4,\n        5,\n        6\n    ]\n}'
    assert candidate({"set": [1, 2, 3]}) == '{\n    "set": [\n        1,\n        2,\n        3\n    ]\n}'
    assert candidate("string") == '"string"'
    assert candidate(None) == 'null'

def test_check():
    check(convert_to_json)
