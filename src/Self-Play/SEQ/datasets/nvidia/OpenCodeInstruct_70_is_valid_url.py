from typing import *

def is_valid_url(url):
    """
    Validates if the given URL is valid according to the specified rules.
    
    Args:
    url (str): The URL to validate.
    
    Returns:
    bool: True if the URL is valid, False otherwise.
    """
    url_pattern = re.compile(r'^(https?:\/\/)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(:\d+)?(\/[^\s]*)?$', re.IGNORECASE)
    return url_pattern.match(url) is not None

### Unit tests below ###
def check(candidate):
    assert candidate("http://example.com") == True
    assert candidate("https://example.com") == True
    assert candidate("example.com") == True
    assert candidate("ftp://example.com") == False
    assert candidate("mailto:test@example.com") == False
    assert candidate("http://sub.domain.co.uk") == True
    assert candidate("https://example.com:8080/path/to/resource") == True
    assert candidate("example.com/path/to/resource") == True
    assert candidate("http://example") == False
    assert candidate("https://example.c") == False

def test_check():
    check(is_valid_url)
