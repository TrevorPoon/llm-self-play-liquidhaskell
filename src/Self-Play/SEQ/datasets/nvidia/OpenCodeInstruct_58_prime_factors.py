from typing import *

def prime_factors(n):
    """
    Computes the prime factors of a given positive integer n and returns them as a list.
    The list contains the prime factors in ascending order, with each factor appearing
    as many times as it divides the number.

    :param n: A positive integer (1 ≤ n ≤ 10^6)
    :return: A list of integers representing the prime factors of n in ascending order.
    """
    factors = []
    divisor = 2
    while n >= 2:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    return factors

### Unit tests below ###
def check(candidate):
    assert candidate(2) == [2]
    assert candidate(3) == [3]
    assert candidate(4) == [2, 2]
    assert candidate(5) == [5]
    assert candidate(6) == [2, 3]
    assert candidate(28) == [2, 2, 7]
    assert candidate(100) == [2, 2, 5, 5]
    assert candidate(101) == [101]
    assert candidate(1) == []
    assert candidate(84) == [2, 2, 3, 7]

def test_check():
    check(prime_factors)
