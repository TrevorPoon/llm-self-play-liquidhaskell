from typing import *

def process_user_data(data):
    """
    Processes user data to return a formatted string with the user's name and age.
    Handles various edge cases including invalid data types, missing keys, and invalid values.
    
    Parameters:
    data (dict): A dictionary containing user information with keys 'name' and 'age'.
    
    Returns:
    str: A formatted string with user details or an error message.
    """
    try:
        if not isinstance(data, dict):
            raise TypeError("Provided data is not a dictionary.")
        
        name = data.get('name')
        age = data.get('age')
        if name is None:
            raise KeyError("Missing 'name' key in dictionary.")
        if age is None:
            raise KeyError("Missing 'age' key in dictionary.")
        
        if not isinstance(name, str) or not name.strip():
            raise ValueError("The 'name' value should be a non-empty string.")
        if not isinstance(age, int):
            raise ValueError("The 'age' value should be an integer.")

        return f"Name: {name}, Age: {age}"
    except (TypeError, ValueError, KeyError) as e:
        return str(e)

### Unit tests below ###
def check(candidate):
    assert candidate("not a dictionary") == "Provided data is not a dictionary."
    assert candidate({'name': '', 'age': 30}) == "The 'name' value should be a non-empty string."
    assert candidate({'name': 'John', 'age': 'invalid'}) == "The 'age' value should be an integer."
    assert candidate({'name': 'Alice'}) == "Missing 'age' key in dictionary."
    assert candidate({'age': 25}) == "Missing 'name' key in dictionary."
    assert candidate({'name': 'Bob', 'age': 23}) == "Name: Bob, Age: 23"
    assert candidate({}) == "Missing 'name' key in dictionary."
    assert candidate({'name': '   ', 'age': 25}) == "The 'name' value should be a non-empty string."
    assert candidate({'name': 'Charlie', 'age': -5}) == "Name: Charlie, Age: -5"
    assert candidate({'name': 'Delta', 'age': 0}) == "Name: Delta, Age: 0"

def test_check():
    check(process_user_data)
