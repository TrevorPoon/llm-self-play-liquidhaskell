from typing import *

def convert_to_pascal_case(snake_str):
    """
    Convert a snake_case string to PascalCase.

    Parameters:
    snake_str (str): A string in snake_case format.

    Returns:
    str: The converted string in PascalCase.
    """
    return ''.join(word.capitalize() for word in snake_str.split('_'))

### Unit tests below ###
def check(candidate):
    assert candidate("example_function_name") == "ExampleFunctionName"
    assert candidate("convert_to_pascal_case") == "ConvertToPascalCase"
    assert candidate("hello_world") == "HelloWorld"
    assert candidate("singleword") == "Singleword"
    assert candidate("multiple__underscores__here") == "MultipleUnderscoresHere"
    assert candidate("leading_underscore") == "LeadingUnderscore"
    assert candidate("trailing_underscore_") == "TrailingUnderscore"
    assert candidate("_both_ends_") == "BothEnds"
    assert candidate("") == ""
    assert candidate("numbers_123_in_string") == "Numbers123InString"

def test_check():
    check(convert_to_pascal_case)
