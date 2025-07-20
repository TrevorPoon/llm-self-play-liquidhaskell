from typing import *

def is_balanced_parentheses(expression):
    """
    Determines if the parentheses in the given expression are balanced.

    :param expression: A string containing only the characters '(', ')', '{', '}', '[' and ']'.
    :return: True if the parentheses are balanced, False otherwise.
    """
    stack = []
    matching_parentheses = {")": "(", "}": "{", "]": "["}

    for char in expression:
        if char in matching_parentheses.values():
            stack.append(char)
        elif char in matching_parentheses:
            if not stack or stack[-1] != matching_parentheses[char]:
                return False
            stack.pop()

    return not stack

### Unit tests below ###
def check(candidate):
    assert candidate("()") == True
    assert candidate("([])") == True
    assert candidate("{[()]}") == True
    assert candidate("([)]") == False
    assert candidate("((()))") == True
    assert candidate(")(") == False
    assert candidate("({[()]})") == True
    assert candidate("({[([)])})") == False
    assert candidate("") == True
    assert candidate("(((((((((())))))))))") == True

def test_check():
    check(is_balanced_parentheses)
