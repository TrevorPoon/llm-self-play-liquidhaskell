from typing import *

def generate_prime_numbers(limit):
    """
    Generate a list of prime numbers up to a given limit using the Sieve of Eratosthenes algorithm.

    :param limit: An integer representing the upper bound up to which prime numbers should be generated.
    :return: A list of integers, where each integer is a prime number less than or equal to the limit.
    """
    if limit < 2:
        return []

    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(limit ** 0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False

    return [num for num, prime in enumerate(is_prime) if prime]

### Unit tests below ###
def check(candidate):
    assert candidate(10) == [2, 3, 5, 7]
    assert candidate(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    assert candidate(1) == []
    assert candidate(0) == []
    assert candidate(2) == [2]
    assert candidate(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert candidate(50) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    assert candidate(100) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    assert candidate(150) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
    assert candidate(2) == [2]

def test_check():
    check(generate_prime_numbers)
