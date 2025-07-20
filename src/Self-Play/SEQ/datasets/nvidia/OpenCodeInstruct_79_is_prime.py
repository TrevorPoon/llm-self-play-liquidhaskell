from typing import *

def is_prime(n):
    """Check if a number is a prime number."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

### Unit tests below ###
def check(candidate):
    assert find_largest_prime([2, 3, 5, 7, 11, 13]) == 13
    assert find_largest_prime([4, 6, 8, 10, 12]) == None
    assert find_largest_prime([17, 19, 23, 29, 31]) == 31
    assert find_largest_prime([1, 0, -1, -3, -5]) == None
    assert find_largest_prime([29, 15, 21, 33, 37]) == 37
    assert find_largest_prime([]) == None
    assert find_largest_prime([2]) == 2
    assert find_largest_prime([3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]) == 97
    assert find_largest_prime([100, 101, 102, 103, 104, 105]) == 103
    assert find_largest_prime([15, 21, 22, 25, 27, 28]) == None

def test_check():
    check(is_prime)
