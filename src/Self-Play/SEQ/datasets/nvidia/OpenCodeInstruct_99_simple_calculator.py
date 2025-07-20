from typing import *

def simple_calculator():
    """
    Simulates a simple calculator that performs arithmetic operations based on user input.
    Handles addition, subtraction, multiplication, and division.
    Outputs an error message for invalid operators or division by zero.
    """
    try:
        num1 = float(input().strip())
        num2 = float(input().strip())
        operator = input().strip()

        if operator == '+':
            result = num1 + num2
            print(f"The result of {num1} + {num2} is {result}")
        elif operator == '-':
            result = num1 - num2
            print(f"The result of {num1} - {num2} is {result}")
        elif operator == '*':
            result = num1 * num2
            print(f"The result of {num1} * {num2} is {result}")
        elif operator == '/':
            if num2 == 0:
                print("Error: Division by zero is not allowed.")
            else:
                result = num1 / num2
                print(f"The result of {num1} / {num2} is {result}")
        else:
            print("Error: Invalid operator.")
    except ValueError:
        print("Error: Invalid input. Please enter numeric values for the numbers.")

### Unit tests below ###
def check(candidate):
    assert candidate(10, 5, '+') == "The result of 10.0 + 5.0 is 15.0"
    assert candidate(10, 5, '-') == "The result of 10.0 - 5.0 is 5.0"
    assert candidate(10, 5, '*') == "The result of 10.0 * 5.0 is 50.0"
    assert candidate(10, 5, '/') == "The result of 10.0 / 5.0 is 2.0"
    assert candidate(10, 0, '/') == "Error: Division by zero is not allowed."
    assert candidate(10, 5, '^') == "Error: Invalid operator."
    assert candidate('a', 5, '+') == "Error: Invalid input. Please enter numeric values for the numbers."
    assert candidate(10, 'b', '-') == "Error: Invalid input. Please enter numeric values for the numbers."
    assert candidate('c', 'd', '*') == "Error: Invalid input. Please enter numeric values for the numbers."
    assert candidate(0, 0, '/') == "Error: Division by zero is not allowed."

def test_check():
    check(simple_calculator)
