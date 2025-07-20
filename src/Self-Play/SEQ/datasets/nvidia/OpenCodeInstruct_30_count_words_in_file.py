from typing import *

def count_words_in_file(filename):
    """
    Counts the number of words in a given text file.

    Parameters:
    filename (str): The name of the file to read.

    Returns:
    int or None: The number of words in the file, or None if the file does not exist.
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()
            words = content.split()
            return len(words)
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")
        return None

### Unit tests below ###
def check(candidate):
    assert candidate("example.txt") == 10  # Assuming example.txt contains exactly 10 words
    assert candidate("nonexistentfile.txt") is None  # File does not exist
    assert candidate("emptyfile.txt") == 0  # Assuming emptyfile.txt is an empty file
    assert candidate("singleword.txt") == 1  # Assuming singleword.txt contains only one word
    assert candidate("multiplelines.txt") == 20  # Assuming multiplelines.txt contains exactly 20 words across multiple lines
    assert candidate("punctuation.txt") == 5  # Assuming punctuation.txt contains "hello, world!" which should count as 2 words
    assert candidate("whitespace.txt") == 3  # Assuming whitespace.txt contains "word1   word2 word3" with irregular spacing
    assert candidate("tabsandspaces.txt") == 4  # Assuming tabsandspaces.txt contains "word1\tword2 word3  word4"
    assert candidate("newlines.txt") == 6  # Assuming newlines.txt contains "word1\nword2\nword3\nword4\nword5\nword6"
    assert candidate("specialchars.txt") == 0  # Assuming specialchars.txt contains only special characters with no words

def test_check():
    check(count_words_in_file)
