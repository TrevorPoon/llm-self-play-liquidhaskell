from typing import *

def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.
    
    Parameters:
    n (int): A non-negative integer whose factorial is to be computed.
    
    Returns:
    int: The factorial of the integer n.
    
    Raises:
    ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

### Unit tests below ###
def check(candidate):
    assert candidate(0) == 1
    assert candidate(1) == 1
    assert candidate(5) == 120
    assert candidate(10) == 3628800
    assert candidate(3) == 6
    assert candidate(7) == 5040
    assert candidate(15) == 1307674368000
    assert candidate(1) == 1
    assert candidate(20) == 2432902008176640000
    assert candidate(100) == 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000

def test_check():
    check(factorial)
