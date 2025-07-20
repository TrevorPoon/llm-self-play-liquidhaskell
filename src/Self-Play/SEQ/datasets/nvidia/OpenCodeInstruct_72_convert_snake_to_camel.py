from typing import *

def convert_snake_to_camel(snake_str):
    """
    Convert a string from snake_case to camelCase format.
    
    Parameters:
    snake_str (str): The input string in snake_case format.
    
    Returns:
    str: The converted string in camelCase format.
    """
    words = snake_str.split('_')
    camel_str = words[0] + ''.join(word.capitalize() for word in words[1:])
    return camel_str

### Unit tests below ###
def check(candidate):
    assert candidate('hello_world') == 'helloWorld'
    assert candidate('this_is_a_test') == 'thisIsATest'
    assert candidate('alreadyCamelCase') == 'alreadyCamelCase'
    assert candidate('PascalCase') == 'pascalCase'
    assert candidate('multiple__underscores__here') == 'multipleUnderscoresHere'
    assert candidate('singleword') == 'singleword'
    assert candidate('_leading_underscore') == '_leadingUnderscore'
    assert candidate('trailing_underscore_') == 'trailingUnderscore_'
    assert candidate('') == ''
    assert candidate('ALL_CAPS_SNAKE_CASE') == 'aLLCapsSnakeCase'

def test_check():
    check(convert_snake_to_camel)
