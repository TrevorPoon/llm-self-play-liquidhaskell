from typing import *

def factorial(n):
    """
    Computes the factorial of a non-negative integer n using recursion.
    
    Args:
        n (int): A non-negative integer.
    
    Returns:
        int: The factorial of the input number n.
    
    Raises:
        ValueError: If n is a negative number or not an integer.
    """
    if not isinstance(n, int):
        raise ValueError("Input must be an integer.")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

### Unit tests below ###
def check(candidate):
    assert candidate(5) == 120
    assert candidate(0) == 1
    assert candidate(1) == 1
    assert candidate(10) == 3628800
    assert candidate(3) == 6
    assert candidate(7) == 5040
    assert candidate(15) == 1307674368000
    try: candidate(-1) except ValueError as e: assert str(e) == "Factorial is not defined for negative numbers."
    try: candidate(3.5) except ValueError as e: assert str(e) == "Input must be an integer."
    try: candidate("5") except ValueError as e: assert str(e) == "Input must be an integer."

def test_check():
    check(factorial)
