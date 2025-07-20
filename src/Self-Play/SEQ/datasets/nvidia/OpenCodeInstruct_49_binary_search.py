from typing import *

def binary_search(arr, target):
    """
    Perform a binary search on a sorted array to find the index of the target element.
    
    Parameters:
    arr (list of int): The sorted array to search.
    target (int): The target element to find.
    
    Returns:
    int: The index of the target element if it exists in the array, otherwise -1.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4, 5], 3) == 2
    assert candidate([1, 2, 3, 4, 5], 1) == 0
    assert candidate([1, 2, 3, 4, 5], 5) == 4
    assert candidate([1, 2, 3, 4, 5], 6) == -1
    assert candidate([], 1) == -1
    assert candidate([1], 1) == 0
    assert candidate([1], 2) == -1
    assert candidate([1, 3, 5, 7, 9, 11], 7) == 3
    assert candidate([1, 3, 5, 7, 9, 11], 12) == -1
    assert candidate([-10, -5, 0, 5, 10], 0) == 2

def test_check():
    check(binary_search)
