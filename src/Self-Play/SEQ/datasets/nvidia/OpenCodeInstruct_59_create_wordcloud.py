from typing import *

def create_wordcloud(text):
    """
    Generates and displays a word cloud from the given text.
    
    Parameters:
    text (str): The input text from which to generate the word cloud.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off axis labels
    plt.show()

### Unit tests below ###
def check(candidate):
    assert candidate("Hello world") is None
    assert candidate("Python is great for data science") is None
    assert candidate("") is None
    assert candidate("A quick brown fox jumps over the lazy dog") is None
    assert candidate("Test test test test test") is None
    assert candidate("123 456 789") is None
    assert candidate("!@#$%^&*()") is None
    assert candidate("SingleWord") is None
    assert candidate("Multiple     spaces    between words") is None
    assert candidate("Mixed CASE Words") is None

def test_check():
    check(create_wordcloud)
