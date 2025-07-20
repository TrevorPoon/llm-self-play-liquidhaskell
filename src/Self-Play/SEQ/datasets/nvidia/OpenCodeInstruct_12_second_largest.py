from typing import *

def second_largest(numbers):
    """
    Returns the second largest unique integer in the list.
    If there are fewer than two unique integers, returns None.
    
    :param numbers: List of integers
    :return: Second largest unique integer or None
    """
    unique_numbers = set(numbers)
    if len(unique_numbers) < 2:
        return None
    unique_numbers.remove(max(unique_numbers))
    return max(unique_numbers)

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4, 5]) == 4
    assert candidate([5, 5, 5, 5]) == None
    assert candidate([10]) == None
    assert candidate([1, 1, 2, 2, 3, 3]) == 2
    assert candidate([100, 90]) == 90
    assert candidate([-1, -2, -3, -4]) == -2
    assert candidate([0, 0, 0, 1]) == 0
    assert candidate([10, 20, 20, 10, 30]) == 20
    assert candidate([1, 2]) == 1
    assert candidate([7, 7, 8, 8, 9, 9]) == 8

def test_check():
    check(second_largest)
