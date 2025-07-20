from typing import *

def sum_of_squares(numbers):
    """
    Calculate the sum of squares of all numbers in the given list.

    :param numbers: List of integers
    :return: Integer sum of squares of the list elements
    """
    return sum(x * x for x in numbers)

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4, 5]) == 55
    assert candidate([-1, -2, -3, -4, -5]) == 55
    assert candidate([0, 0, 0, 0, 0]) == 0
    assert candidate([10, 20, 30]) == 1400
    assert candidate([100, 200, 300, 400]) == 300000
    assert candidate([]) == 0
    assert candidate([1]) == 1
    assert candidate([-1]) == 1
    assert candidate([1.5, 2.5, 3.5]) == 21.25
    assert candidate([-1.5, -2.5, -3.5]) == 21.25

def test_check():
    check(sum_of_squares)
