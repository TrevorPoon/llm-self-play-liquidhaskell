from typing import *

def list_directory_contents(path):
    """
    Lists all files and subdirectories within the specified directory.
    
    Parameters:
    path (str): The path to the directory to be listed.
    
    Returns:
    None
    """
    if not os.path.exists(path):
        print(f"Error: The directory '{path}' does not exist.")
        return
    
    try:
        contents = os.listdir(path)
        print(f"Contents of '{path}':")
        for item in contents:
            print(item)
    except Exception as e:
        print(f"An error occurred: {e}")

### Unit tests below ###
def check(candidate):
    assert candidate("/nonexistent_directory") is None
    assert candidate("") is None
    assert candidate("/") is None
    assert candidate(os.getcwd()) is None
    assert candidate(os.path.dirname(os.path.abspath(__file__))) is None
    assert candidate("/tmp") is None
    assert candidate("/usr") is None
    assert candidate("/var/log") is None
    assert candidate(os.path.expanduser("~")) is None
    assert candidate(os.path.join(os.path.expanduser("~"), "Documents")) is None

def test_check():
    check(list_directory_contents)
