from typing import *

def calculate_sum_of_squares(numbers):
    """
    Calculate the sum of squares of a list of floating-point numbers.
    
    :param numbers: List of floating-point numbers.
    :return: Sum of the squares of the numbers.
    """
    return sum(x * x for x in numbers)

### Unit tests below ###
def check(candidate):
    assert candidate([1.0, 2.0, 3.0]) == 14.0
    assert candidate([0.0, 0.0, 0.0]) == 0.0
    assert candidate([-1.0, -2.0, -3.0]) == 14.0
    assert candidate([1.5, 2.3, 3.7, 4.1]) == 34.899999999999996
    assert candidate([1e-10, 2e-10, 3e-10]) == 1.4e-19
    assert candidate([1.1, 1.1, 1.1, 1.1]) == 4.840000000000001
    assert candidate([1000000.0, 2000000.0]) == 5000000000000.0
    assert candidate([]) == 0.0
    assert candidate([1.23456789, 9.87654321]) == 98.38699999999999
    assert candidate([1.0, -1.0, 1.0, -1.0]) == 4.0

def test_check():
    check(calculate_sum_of_squares)
