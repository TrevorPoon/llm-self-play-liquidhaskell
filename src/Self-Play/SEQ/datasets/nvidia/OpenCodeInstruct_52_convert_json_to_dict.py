from typing import *

def convert_json_to_dict(filename):
    """
    Reads a JSON file and converts it into a nested dictionary.

    Parameters:
    filename (str): The name of the JSON file to be read.

    Returns:
    dict: The nested dictionary representation of the JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

### Unit tests below ###
def check(candidate):
    assert candidate("test1.json") == {"name": "John", "age": 30, "city": "New York"}
    assert candidate("test2.json") == {"employees": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]}
    assert candidate("test3.json") == {"fruits": ["apple", "banana", "cherry"], "vegetables": ["carrot", "broccoli"]}
    assert candidate("test4.json") == {"person": {"name": "Charlie", "address": {"street": "123 Main St", "city": "Los Angeles"}}}
    assert candidate("test5.json") == {"numbers": [1, 2, 3, 4, 5], "letters": ["a", "b", "c"]}
    assert candidate("test6.json") == {"empty": {}}
    assert candidate("test7.json") == {"single_key": "single_value"}
    assert candidate("test8.json") == {"nested": {"level1": {"level2": {"level3": "value"}}}}
    assert candidate("test9.json") == {"array_of_objects": [{"id": 1, "value": "one"}, {"id": 2, "value": "two"}]}
    assert candidate("test10.json") == {"boolean_values": {"true": True, "false": False}}

def test_check():
    check(convert_json_to_dict)
