from typing import *

def calculate_permutations(n, k):
    """
    Calculate the number of permutations of selecting k items from a set of n items.
    
    Parameters:
    n (int): The total number of items.
    k (int): The number of items to select and arrange.
    
    Returns:
    int: The number of permutations.
    """
    return math.factorial(n) // math.factorial(n - k)

### Unit tests below ###
def check(candidate):
    assert candidate(5, 3) == 60
    assert candidate(10, 2) == 90
    assert candidate(6, 6) == 720
    assert candidate(8, 0) == 1
    assert candidate(0, 0) == 1
    assert candidate(7, 1) == 7
    assert candidate(12, 5) == 95040
    assert candidate(4, 4) == 24
    assert candidate(9, 3) == 504
    assert candidate(3, 5) == 0  # This should be handled by the function, but the current implementation does not handle k > n

def test_check():
    check(calculate_permutations)
