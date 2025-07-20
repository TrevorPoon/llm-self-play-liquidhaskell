from typing import *

def find_median(numbers):
    """
    Computes the median of a list of integers.
    
    Parameters:
    numbers (list of int): The list of integers for which to find the median.
    
    Returns:
    float: The median of the list.
    """
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 1:
        return float(sorted_numbers[n // 2])
    else:
        mid1, mid2 = sorted_numbers[n // 2 - 1], sorted_numbers[n // 2]
        return (mid1 + mid2) / 2.0

### Unit tests below ###
def check(candidate):
    assert candidate([3, 1, 4, 1, 5, 9, 2]) == 3
    assert candidate([3, 1, 4, 1, 5, 9]) == 3.5
    assert candidate([1, 2, 3, 4, 5]) == 3
    assert candidate([10, 20, 30, 40, 50, 60]) == 35.0
    assert candidate([7]) == 7
    assert candidate([1, 1, 1, 1, 1, 1]) == 1.0
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 5.5
    assert candidate([-5, -1, -3, -2, -4]) == -3
    assert candidate([1.5, 2.5, 3.5, 4.5, 5.5]) == 3.5
    assert candidate([100, 200]) == 150.0

def test_check():
    check(find_median)
