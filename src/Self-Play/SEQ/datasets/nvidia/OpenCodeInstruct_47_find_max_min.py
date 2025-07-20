from typing import *

def find_max_min(lst):
    """
    Finds the maximum and minimum values in a list of integers.

    Parameters:
    lst (list): A list of integers.

    Returns:
    tuple: A tuple containing the maximum and minimum values, or None if the list is empty.
    """
    if not lst:
        return None
    
    max_val = lst[0]
    min_val = lst[0]
    
    for num in lst:
        if num > max_val:
            max_val = num
        if num < min_val:
            min_val = num
    
    return (max_val, min_val)

### Unit tests below ###
def check(candidate):
    assert candidate([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) == (9, 1)
    assert candidate([10, 20, 30, 40, 50]) == (50, 10)
    assert candidate([-10, -20, -30, -40, -50]) == (-10, -50)
    assert candidate([0, 0, 0, 0, 0]) == (0, 0)
    assert candidate([100]) == (100, 100)
    assert candidate([-100]) == (-100, -100)
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == (10, 1)
    assert candidate([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == (10, 1)
    assert candidate([]) == None
    assert candidate([5, 5, 5, 5, 5, 5]) == (5, 5)

def test_check():
    check(find_max_min)
