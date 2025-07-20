from typing import *

def count_odd_even_digits(n):
    """
    Returns a dictionary with the count of odd and even digits in the given integer n.
    
    Parameters:
    n (int): The integer to analyze.
    
    Returns:
    dict: A dictionary with keys 'odd' and 'even' representing the counts of odd and even digits.
    """
    if n == 0:
        return {'odd': 0, 'even': 1}
    
    counts = {'odd': 0, 'even': 0}
    n = abs(n)
    
    while n > 0:
        digit = n % 10
        if digit % 2 == 0:
            counts['even'] += 1
        else:
            counts['odd'] += 1
        n = n // 10
    
    return counts

### Unit tests below ###
def check(candidate):
    assert candidate(0) == {'odd': 0, 'even': 1}
    assert candidate(123456) == {'odd': 3, 'even': 3}
    assert candidate(111111) == {'odd': 6, 'even': 0}
    assert candidate(24680) == {'odd': 0, 'even': 5}
    assert candidate(-987654321) == {'odd': 5, 'even': 4}
    assert candidate(13579) == {'odd': 5, 'even': 0}
    assert candidate(222222) == {'odd': 0, 'even': 6}
    assert candidate(101010) == {'odd': 3, 'even': 3}
    assert candidate(9876543210) == {'odd': 5, 'even': 5}
    assert candidate(-1) == {'odd': 1, 'even': 0}

def test_check():
    check(count_odd_even_digits)
