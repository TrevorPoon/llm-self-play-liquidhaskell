from typing import *

def download_html(url):
    """
    Downloads and returns the HTML content of the specified URL.
    
    Args:
    url (str): The URL of the web page to download.
    
    Returns:
    str: The HTML content of the web page.
    
    Raises:
    requests.exceptions.HTTPError: If the HTTP request returns an unsuccessful status code.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    return response.text

### Unit tests below ###
def check(candidate):
    assert candidate("https://httpbin.org/status/200").startswith("<!DOCTYPE html>")
    assert candidate("https://httpbin.org/status/200").find("HTTPBin") != -1
    assert candidate("https://httpbin.org/status/404").startswith("<!DOCTYPE html>")
    assert candidate("https://httpbin.org/status/404").find("404 Not Found") != -1
    assert candidate("https://httpbin.org/status/500").startswith("<!DOCTYPE html>")
    assert candidate("https://httpbin.org/status/500").find("500 Internal Server Error") != -1
    assert candidate("https://httpbin.org/get").find("args") != -1
    assert candidate("https://httpbin.org/get").find("headers") != -1
    assert candidate("https://httpbin.org/get").find("origin") != -1
    assert candidate("https://httpbin.org/get").find("url") != -1

def test_check():
    check(download_html)
