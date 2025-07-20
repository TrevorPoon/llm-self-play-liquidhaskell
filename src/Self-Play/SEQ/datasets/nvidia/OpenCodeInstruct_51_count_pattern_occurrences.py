from typing import *

def count_pattern_occurrences(text, pattern):
    """
    Counts the number of times the pattern appears in the text, including overlapping occurrences.

    :param text: The string in which to search for the pattern.
    :param pattern: The string pattern to search for in the text.
    :return: The number of times the pattern appears in the text.
    """
    count = 0
    pattern_length = len(pattern)
    for i in range(len(text) - pattern_length + 1):
        if text[i:i + pattern_length] == pattern:
            count += 1
    return count

### Unit tests below ###
def check(candidate):
    assert candidate('abcabcabc', 'abc') == 3
    assert candidate('aaaa', 'aa') == 3
    assert candidate('ababab', 'aba') == 2
    assert candidate('hello world', 'o') == 2
    assert candidate('mississippi', 'issi') == 1
    assert candidate('aaaaa', 'a') == 5
    assert candidate('abc', 'abcd') == 0
    assert candidate('', 'a') == 0
    assert candidate('abc', '') == 0
    assert candidate('', '') == 0

def test_check():
    check(count_pattern_occurrences)
