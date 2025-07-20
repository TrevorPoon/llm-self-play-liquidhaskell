from typing import *

class FileManager:
    def read_file(self, file_path):
        """Read the contents of a file and return it as a string. Return an empty string if the file does not exist."""
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return ''

    def write_file(self, file_path, content):
        """Write the provided content to a file, overwriting the file if it already exists."""
        with open(file_path, 'w') as file:
            file.write(content)

    def append_to_file(self, file_path, content):
        """Append the provided content to the end of a file. Create the file if it does not exist."""
        with open(file_path, 'a') as file:
            file.write(content)

    def delete_file(self, file_path):
        """Delete the specified file. Do nothing if the file does not exist."""
        if os.path.exists(file_path):
            os.remove(file_path)

### Unit tests below ###
def check(candidate):
    assert candidate.write_file('test.txt', 'Hello, world!') is None
    assert candidate.read_file('test.txt') == 'Hello, world!'
    assert candidate.append_to_file('test.txt', ' Welcome!') is None
    assert candidate.read_file('test.txt') == 'Hello, world! Welcome!'
    assert candidate.delete_file('test.txt') is None
    assert os.path.exists('test.txt') == False
    candidate.write_file('test2.txt', 'Line 1\nLine 2\nLine 3')
    assert candidate.read_file('test2.txt') == 'Line 1\nLine 2\nLine 3'
    candidate.append_to_file('test2.txt', '\nLine 4')
    assert candidate.read_file('test2.txt') == 'Line 1\nLine 2\nLine 3\nLine 4'
    candidate.delete_file('test2.txt')
    assert os.path.exists('test2.txt') == False
    candidate.write_file('test3.txt', 'Python')
    candidate.append_to_file('test3.txt', ' is fun!')
    assert candidate.read_file('test3.txt') == 'Python is fun!'

def test_check():
    check(FileManager())
