from typing import *

def sum_even_odd(numbers):
    """
    Computes the sum of even and odd numbers in a list.

    Parameters:
    numbers (list of int): A list of integers.

    Returns:
    tuple: A tuple containing the sum of even numbers and the sum of odd numbers.
    """
    even_sum = sum(num for num in numbers if num % 2 == 0)
    odd_sum = sum(num for num in numbers if num % 2 != 0)
    return (even_sum, odd_sum)

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4, 5]) == (6, 9)
    assert candidate([2, 4, 6, 8, 10]) == (30, 0)
    assert candidate([1, 3, 5, 7, 9]) == (0, 25)
    assert candidate([]) == (0, 0)
    assert candidate([0]) == (0, 0)
    assert candidate([-2, -4, -6, -8, -10]) == (-30, 0)
    assert candidate([-1, -3, -5, -7, -9]) == (0, -25)
    assert candidate([-1, 2, -3, 4, -5, 6]) == (12, -9)
    assert candidate([100, 200, 300, 1, 3, 5]) == (600, 9)
    assert candidate([11, 22, 33, 44, 55, 66]) == (132, 99)

def test_check():
    check(sum_even_odd)
