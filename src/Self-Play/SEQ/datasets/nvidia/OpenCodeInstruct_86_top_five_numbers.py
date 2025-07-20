from typing import *

def top_five_numbers(numbers):
    """
    Returns the top 5 largest numbers from the list, sorted in descending order.
    If the list contains fewer than 5 numbers, returns the entire list sorted in descending order.

    :param numbers: List of integers
    :return: List of top 5 largest integers sorted in descending order
    """
    sorted_numbers = sorted(numbers, reverse=True)
    return sorted_numbers if len(sorted_numbers) < 5 else sorted_numbers[:5]

### Unit tests below ###
def check(candidate):
    assert candidate([10, 20, 30, 40, 50]) == [50, 40, 30, 20, 10]
    assert candidate([5, 1, 9, 3, 7, 6, 8, 2, 4, 0]) == [9, 8, 7, 6, 5]
    assert candidate([100]) == [100]
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [10, 9, 8, 7, 6]
    assert candidate([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) == [100, 90, 80, 70, 60]
    assert candidate([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]) == [5, 5, 5, 5, 5]
    assert candidate([-1, -2, -3, -4, -5]) == [-1, -2, -3, -4, -5]
    assert candidate([10, 20, 30]) == [30, 20, 10]
    assert candidate([]) == []
    assert candidate([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20]) == [100, 90, 80, 70, 60]

def test_check():
    check(top_five_numbers)
