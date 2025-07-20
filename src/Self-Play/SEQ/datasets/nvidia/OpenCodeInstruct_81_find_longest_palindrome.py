from typing import *

def find_longest_palindrome(s):
    """
    Finds the longest palindromic substring in the given string s.
    
    :param s: Input string
    :return: Longest palindromic substring
    """
    if not s:
        return ""
    
    start, end = 0, 0
    
    for i in range(len(s)):
        len1 = expand_around_center(s, i, i)
        len2 = expand_around_center(s, i, i + 1)
        max_len = max(len1, len2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]

### Unit tests below ###
def check(candidate):
    assert candidate("babad") == "bab"
    assert candidate("cbbd") == "bb"
    assert candidate("a") == "a"
    assert candidate("ac") == "a"
    assert candidate("racecar") == "racecar"
    assert candidate("noon high it is") == "noon"
    assert candidate("abacdfgdcaba") == "aba"
    assert candidate("aaaa") == "aaaa"
    assert candidate("") == ""
    assert candidate("abcba") == "abcba"

def test_check():
    check(find_longest_palindrome)
