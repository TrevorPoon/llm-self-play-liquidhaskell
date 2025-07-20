from typing import *

def bubble_sort(arr):
    """
    Sorts a list of integers in ascending order using the bubble sort algorithm.

    Parameters:
    arr (list): The list of integers to be sorted.

    Returns:
    None: The list is sorted in place.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

### Unit tests below ###
def check(candidate):
    assert candidate([64, 34, 25, 12, 22, 11, 90]) is None
    assert candidate([64, 34, 25, 12, 22, 11, 90]) == [11, 12, 22, 25, 34, 64, 90]
    assert candidate([5, 1, 4, 2, 8]) == [1, 2, 4, 5, 8]
    assert candidate([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert candidate([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert candidate([]) == []
    assert candidate([1]) == [1]
    assert candidate([2, 1]) == [1, 2]
    assert candidate([3, 3, 3, 3, 3]) == [3, 3, 3, 3, 3]
    assert candidate([-1, -2, -3, -4, -5]) == [-5, -4, -3, -2, -1]

def test_check():
    check(bubble_sort)
