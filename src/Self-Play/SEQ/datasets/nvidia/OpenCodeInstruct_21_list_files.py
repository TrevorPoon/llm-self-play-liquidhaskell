from typing import *

def list_files(directory_path):
    """
    Recursively lists all files in the given directory and its subdirectories.
    
    Args:
    directory_path (str): The path to the directory to be searched.
    
    Returns:
    None: Prints the absolute path of each file found.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            print(os.path.join(root, file))

### Unit tests below ###
def check(candidate):
    assert candidate("/nonexistent_directory") is None
    assert candidate("") is None
    assert candidate("/") is None
    assert candidate("/tmp") is None
    assert candidate("/etc") is None
    assert candidate("/var/log") is None
    assert candidate("/usr/bin") is None
    assert candidate("/home") is None
    assert candidate("/root") is None
    assert candidate("/dev") is None

def test_check():
    check(list_files)
