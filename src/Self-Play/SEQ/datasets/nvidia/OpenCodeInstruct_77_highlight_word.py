from typing import *

def highlight_word(text, word):
    """
    Highlights all occurrences of a specific word in a given text by surrounding it with asterisks (*).
    
    Parameters:
    text (str): The input text where the word needs to be highlighted.
    word (str): The word to be highlighted in the text.
    
    Returns:
    str: The modified text with the specified word highlighted.
    """
    highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', '*' + word + '*', text)
    return highlighted_text

### Unit tests below ###
def check(candidate):
    assert candidate("Python is great", "Python") == "*Python* is great"
    assert candidate("Python is great, python is fun", "Python") == "*Python* is great, python is fun"
    assert candidate("python is great, Python is fun", "python") == "*python* is great, *Python* is fun"
    assert candidate("PythonPython", "Python") == "PythonPython"
    assert candidate("Python is great. Python.", "Python") == "*Python* is great. *Python*."
    assert candidate("Python", "Python") == "*Python*"
    assert candidate("Python is great", "python") == "Python is great"
    assert candidate("Python is great", "is") == "Python *is* great"
    assert candidate("Python is great", "Python is") == "Python is great"
    assert candidate("", "Python") == ""

def test_check():
    check(highlight_word)
