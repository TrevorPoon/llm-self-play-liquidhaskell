from typing import *

def largest_prime_factor(n):
    """
    Returns the largest prime factor of the given number n.
    If n is less than 2, returns None.
    
    :param n: Integer, the number to find the largest prime factor of.
    :return: Integer, the largest prime factor of n, or None if n < 2.
    """
    if n < 2:
        return None

    def is_prime(num):
        """
        Checks if a number is prime.
        
        :param num: Integer, the number to check for primality.
        :return: Boolean, True if num is prime, False otherwise.
        """
        if num < 2:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

    largest_factor = None

    while n % 2 == 0:
        largest_factor = 2
        n //= 2

    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            if is_prime(i):
                largest_factor = i
            n //= i

    if n > 2:
        largest_factor = n

    return largest_factor

### Unit tests below ###
def check(candidate):
    assert candidate(13195) == 29
    assert candidate(600851475143) == 6857
    assert candidate(2) == 2
    assert candidate(3) == 3
    assert candidate(4) == 2
    assert candidate(9) == 3
    assert candidate(15) == 5
    assert candidate(1) == None
    assert candidate(0) == None
    assert candidate(-10) == None

def test_check():
    check(largest_prime_factor)
