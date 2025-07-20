from typing import *

def find_two_sum_indices(numbers, target):
    """
    Finds two distinct indices in the list 'numbers' such that the numbers at these indices add up to 'target'.
    
    :param numbers: List of integers.
    :param target: Integer target sum.
    :return: List of two indices if a pair is found, otherwise an empty list.
    """
    num_to_index = {}
    for index, num in enumerate(numbers):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], index]
        num_to_index[num] = index
    return []

### Unit tests below ###
def check(candidate):
    assert candidate([2, 7, 11, 15], 9) == [0, 1]
    assert candidate([3, 2, 4], 6) == [1, 2]
    assert candidate([3, 3], 6) == [0, 1]
    assert candidate([1, 2, 3, 4, 5], 10) == [3, 4]
    assert candidate([1, 2, 3, 4, 5], 8) == [2, 4]
    assert candidate([1, 2, 3, 4, 5], 11) == []
    assert candidate([], 0) == []
    assert candidate([1], 1) == []
    assert candidate([0, 4, 3, 0], 0) == [0, 3]
    assert candidate([-1, -2, -3, -4, -5], -8) == [2, 4]

def test_check():
    check(find_two_sum_indices)
