from typing import *

def find_most_frequent_word(text):
    """
    Finds the most frequently occurring word in the given text.
    
    Parameters:
    text (str): The input text containing words separated by spaces and punctuation.
    
    Returns:
    str: The most frequently occurring word in the text.
    """
    from collections import Counter
    import re
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    word_counts = Counter(words)
    most_frequent_word, _ = word_counts.most_common(1)[0]
    
    return most_frequent_word

### Unit tests below ###
def check(candidate):
    assert candidate("This is a test. This test is only a test.") == "test"
    assert candidate("Hello world! Hello everyone.") == "hello"
    assert candidate("One fish two fish red fish blue fish.") == "fish"
    assert candidate("To be or not to be, that is the question.") == "to"
    assert candidate("A quick brown fox jumps over the lazy dog.") == "a"
    assert candidate("Python python PYTHON pyThOn.") == "python"
    assert candidate("Unique words only here.") == "unique"
    assert candidate("") == ""
    assert candidate("123 456 789 123 456 789 123") == ""
    assert candidate("Repeat repeat repeat repeat.") == "repeat"

def test_check():
    check(find_most_frequent_word)
