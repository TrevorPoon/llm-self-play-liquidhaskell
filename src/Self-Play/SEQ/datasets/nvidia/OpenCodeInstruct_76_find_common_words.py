from typing import *

def find_common_words(text1, text2):
    """
    Returns a list of common words found in both input strings, ignoring case and stop words.
    
    :param text1: First input string
    :param text2: Second input string
    :return: List of common words
    """
    # Define a set of common stop words
    stop_words = set(['the', 'and', 'is', 'a', 'an', 'of', 'for', 'it', 'in', 'on', 'at', 'to', 'with', 'as', 'by', 'that', 'this', 'these', 'those'])

    # Helper function to clean and split text into words
    def clean_text(text):
        # Use regex to keep only alphanumeric characters and spaces, then lower case the text
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
        # Split the text into words
        words = cleaned_text.split()
        # Filter out stop words
        filtered_words = [word for word in words if word not in stop_words]
        return filtered_words

    # Get words from both texts
    words1 = clean_text(text1)
    words2 = clean_text(text2)

    # Count word occurrences in both lists
    count1 = Counter(words1)
    count2 = Counter(words2)

    # Find common words
    common_words = list((count1 & count2).elements())

    return common_words

### Unit tests below ###
def check(candidate):
    assert candidate("Python is great", "Python is fun") == ["python", "is"]
    assert candidate("Hello world", "world of code") == ["world"]
    assert candidate("The quick brown fox", "The lazy brown dog") == ["the", "brown"]
    assert candidate("Data science and data analysis", "Data analysis is key") == ["data", "analysis"]
    assert candidate("This is a test", "This is only a test") == ["this", "is", "a", "test"]
    assert candidate("One fish two fish", "Red fish blue fish") == ["fish"]
    assert candidate("Stop words are the and is", "Remove stop words like the and is") == ["stop", "words", "like"]
    assert candidate("No common words here", "Completely different") == []
    assert candidate("Special $characters #should &be *ignored!", "Characters should be ignored") == ["characters", "should", "be", "ignored"]
    assert candidate("UPPER and lower CASE", "Case should be ignored") == ["case", "should", "be", "ignored"]

def test_check():
    check(find_common_words)
