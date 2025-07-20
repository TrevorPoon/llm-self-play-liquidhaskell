from typing import *

def string_permutations(input_string):
    """
    Generate all unique permutations of the input string and return them as a sorted list.

    :param input_string: A string for which permutations are to be generated.
    :return: A sorted list of unique permutations of the input string.
    """
    perm = set(permutations(input_string))
    result = sorted([''.join(p) for p in perm])
    return result

### Unit tests below ###
def check(candidate):
    assert candidate("aab") == ["aab", "aba", "baa"]
    assert candidate("abc") == ["abc", "acb", "bac", "bca", "cab", "cba"]
    assert candidate("a") == ["a"]
    assert candidate("") == []
    assert candidate("aaa") == ["aaa"]
    assert candidate("ab") == ["ab", "ba"]
    assert candidate("aabb") == ["aabb", "abab", "abba", "baab", "baba", "bbaa"]
    assert candidate("123") == ["123", "132", "213", "231", "312", "321"]
    assert candidate("!@#") == ["!@#", "!#@#", "@!#", "@#@!", "#!@", "#@!"]
    assert candidate("zzz") == ["zzz"]

def test_check():
    check(string_permutations)
