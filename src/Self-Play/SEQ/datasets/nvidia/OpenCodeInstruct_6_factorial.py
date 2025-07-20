from typing import *

def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.
    
    Parameters:
    n (int): A non-negative integer whose factorial is to be computed.
    
    Returns:
    int: The factorial of the given integer n.
    
    Raises:
    ValueError: If n is a negative integer.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

### Unit tests below ###
def check(candidate):
    assert candidate(0) == 1
    assert candidate(1) == 1
    assert candidate(2) == 2
    assert candidate(3) == 6
    assert candidate(4) == 24
    assert candidate(5) == 120
    assert candidate(6) == 720
    assert candidate(10) == 3628800
    assert candidate(12) == 479001600
    assert candidate(15) == 1307674368000

def test_check():
    check(factorial)
