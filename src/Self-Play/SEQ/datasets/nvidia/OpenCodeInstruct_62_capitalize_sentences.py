from typing import *

def capitalize_sentences(sentences):
    """
    Capitalizes the first letter of each word in each sentence of the input list and converts the rest of the letters to lowercase.

    Parameters:
    sentences (list of str): A list of sentences where each sentence is a string.

    Returns:
    list of str: A new list of sentences with each word capitalized appropriately.
    """
    return [' '.join(word.capitalize() for word in sentence.split()) for sentence in sentences]

### Unit tests below ###
def check(candidate):
    assert candidate(["hello world"]) == ["Hello World"]
    assert candidate(["python is great", "i love coding"]) == ["Python Is Great", "I Love Coding"]
    assert candidate(["THIS IS A TEST", "another TEST sentence"]) == ["This Is A Test", "Another Test Sentence"]
    assert candidate([""]) == [""]
    assert candidate(["multiple    spaces"]) == ["Multiple    Spaces"]
    assert candidate(["123abc def456"]) == ["123Abc Def456"]
    assert candidate(["!@# $%^ &*("]) == ["!@# $%^ &*("]
    assert candidate(["capitalize THIS sentence", "and THIS one"]) == ["Capitalize This Sentence", "And This One"]
    assert candidate(["singleword", "twowords"]) == ["Singleword", "Twowords"]
    assert candidate(["mixed CASE with NUMBERS 123"]) == ["Mixed Case With Numbers 123"]

def test_check():
    check(capitalize_sentences)
