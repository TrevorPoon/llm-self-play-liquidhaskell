from typing import *

def format_person_info(info):
    """
    Formats the given dictionary containing a person's information into a string.

    Parameters:
    info (dict): A dictionary with keys 'name', 'age', and 'city'.

    Returns:
    str: A formatted string in the form "Name: [name], Age: [age], City: [city]".
    """
    return f"Name: {info['name']}, Age: {info['age']}, City: {info['city']}"

### Unit tests below ###
def check(candidate):
    assert candidate({'name': 'John Doe', 'age': 30, 'city': 'New York'}) == "Name: John Doe, Age: 30, City: New York"
    assert candidate({'name': 'Jane Smith', 'age': 25, 'city': 'Chicago'}) == "Name: Jane Smith, Age: 25, City: Chicago"
    assert candidate({'name': 'Emily Davis', 'age': 40, 'city': 'San Francisco'}) == "Name: Emily Davis, Age: 40, City: San Francisco"
    assert candidate({'name': 'Michael Brown', 'age': 35, 'city': 'Seattle'}) == "Name: Michael Brown, Age: 35, City: Seattle"
    assert candidate({'name': 'Sarah Wilson', 'age': 22, 'city': 'Boston'}) == "Name: Sarah Wilson, Age: 22, City: Boston"
    assert candidate({'name': 'David Lee', 'age': 50, 'city': 'Miami'}) == "Name: David Lee, Age: 50, City: Miami"
    assert candidate({'name': 'Olivia Martinez', 'age': 29, 'city': 'Dallas'}) == "Name: Olivia Martinez, Age: 29, City: Dallas"
    assert candidate({'name': 'Daniel Hernandez', 'age': 45, 'city': 'Austin'}) == "Name: Daniel Hernandez, Age: 45, City: Austin"
    assert candidate({'name': 'Sophia Garcia', 'age': 33, 'city': 'Denver'}) == "Name: Sophia Garcia, Age: 33, City: Denver"
    assert candidate({'name': 'James Lopez', 'age': 27, 'city': 'Philadelphia'}) == "Name: James Lopez, Age: 27, City: Philadelphia"

def test_check():
    check(format_person_info)
