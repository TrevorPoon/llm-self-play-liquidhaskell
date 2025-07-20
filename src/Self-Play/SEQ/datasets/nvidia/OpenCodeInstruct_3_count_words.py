from typing import *

def count_words(text):
    """
    Returns a dictionary with the frequency of each word in the input string,
    excluding common stop words.

    :param text: A string consisting of lowercase letters and spaces.
    :return: A dictionary with words as keys and their frequencies as values.
    """
    # List of stop words to ignore
    stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with'}
    
    # Split the text into words
    words = text.split()
    
    # Dictionary to store the frequency of each word
    word_count = {}
    
    # Iterate over each word in the list
    for word in words:
        # If the word is not a stop word
        if word not in stop_words:
            # Increment the word count
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    
    return word_count

### Unit tests below ###
def check(candidate):
    assert candidate("the quick brown fox jumps over the lazy dog") == {'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
    assert candidate("a quick brown fox jumps over the lazy dog") == {'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
    assert candidate("the the the the") == {}
    assert candidate("hello world hello") == {'hello': 2, 'world': 1}
    assert candidate("with the and for in on at by") == {}
    assert candidate("python is great and python is fun") == {'python': 2, 'is': 2, 'great': 1, 'fun': 1}
    assert candidate("this is a test") == {'this': 1, 'test': 1}
    assert candidate("an example with some stop words") == {'example': 1, 'with': 1, 'some': 1, 'stop': 1, 'words': 1}
    assert candidate("one two three four five six seven eight nine ten") == {'one': 1, 'two': 1, 'three': 1, 'four': 1, 'five': 1, 'six': 1, 'seven': 1, 'eight': 1, 'nine': 1, 'ten': 1}
    assert candidate("") == {}

def test_check():
    check(count_words)
