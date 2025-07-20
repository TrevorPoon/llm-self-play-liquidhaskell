from typing import *

def remove_chars_at_positions(text, positions):
    """
    Removes characters from the string at the specified positions.

    :param text: The original string from which characters will be removed.
    :param positions: A list of integer positions indicating which characters to remove.
    :return: A new string with the specified characters removed.
    """
    positions_set = set(positions)
    return ''.join(char for idx, char in enumerate(text) if idx not in positions_set)

### Unit tests below ###
def check(candidate):
    assert candidate("hello world", [0, 5, 10]) == "ello orld"
    assert candidate("abcdef", [1, 3, 5]) == "ace"
    assert candidate("abcdef", []) == "abcdef"
    assert candidate("abcdef", [0, 1, 2, 3, 4, 5]) == ""
    assert candidate("abcdef", [6]) == "abcdef"
    assert candidate("abcdef", [-1]) == "abcdef"
    assert candidate("", [0]) == ""
    assert candidate("a", [0]) == ""
    assert candidate("a", [1]) == "a"
    assert candidate("hello world", [2, 4, 6, 8]) == "hl olrld"

def test_check():
    check(remove_chars_at_positions)
