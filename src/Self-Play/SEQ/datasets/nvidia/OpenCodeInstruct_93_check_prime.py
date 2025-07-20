from typing import *

def check_prime(n):
    """Return True if n is a prime number, False otherwise."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

### Unit tests below ###
def check(candidate):
    assert candidate(1) == False
    assert candidate(2) == True
    assert candidate(3) == True
    assert candidate(4) == False
    assert candidate(29) == True
    assert candidate(179) == True
    assert candidate(180) == False
    assert candidate(0) == False
    assert candidate(-5) == False
    assert candidate(97) == True

def test_check():
    check(check_prime)
