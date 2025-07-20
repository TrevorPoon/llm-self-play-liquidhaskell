from typing import *

def remove_duplicates(input_list):
    """
    Removes all duplicate elements from the input list while preserving the order of their first occurrence.

    :param input_list: List of integers from which duplicates need to be removed.
    :return: A new list with duplicates removed.
    """
    result = []
    seen = set()
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 2, 1, 4, 5]) == [1, 2, 3, 4, 5]
    assert candidate([10, 20, 20, 10, 30]) == [10, 20, 30]
    assert candidate([1, 1, 1, 1, 1]) == [1]
    assert candidate([]) == []
    assert candidate([5, 4, 3, 2, 1]) == [5, 4, 3, 2, 1]
    assert candidate([7, 8, 9, 7, 10, 9, 8]) == [7, 8, 9, 10]
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert candidate([100, 200, 100, 200, 100]) == [100, 200]
    assert candidate([-1, -2, -3, -2, -1]) == [-1, -2, -3]
    assert candidate([0, 0, 0, 0, 0]) == [0]

def test_check():
    check(remove_duplicates)
