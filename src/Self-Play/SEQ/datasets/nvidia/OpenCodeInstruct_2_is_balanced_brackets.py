from typing import *

def is_balanced_brackets(expression):
    """
    Checks if all the brackets in the given expression are balanced.
    
    :param expression: A string containing characters including parentheses, square brackets, and curly braces.
    :return: True if the brackets are balanced, False otherwise.
    """
    stack = []
    bracket_map = {')': '(', ']': '[', '}': '{'}
    
    for char in expression:
        if char in bracket_map.values():
            stack.append(char)
        elif char in bracket_map:
            if not stack or bracket_map[char] != stack.pop():
                return False
    
    return not stack

### Unit tests below ###
def check(candidate):
    assert candidate("()") == True
    assert candidate("([])") == True
    assert candidate("{[()]}") == True
    assert candidate("{[(])}") == False
    assert candidate("([)]") == False
    assert candidate("((()))") == True
    assert candidate("(()") == False
    assert candidate("") == True
    assert candidate("{[}]") == False
    assert candidate("a(b)c[d]{e}") == True

def test_check():
    check(is_balanced_brackets)
