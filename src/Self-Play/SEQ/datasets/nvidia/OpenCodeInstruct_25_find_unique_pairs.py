from typing import *

def find_unique_pairs(nums, target_sum):
    """
    Finds all unique pairs of numbers in the list that add up to the target sum.
    
    :param nums: List of integers.
    :param target_sum: Integer representing the target sum.
    :return: List of unique pairs, where each pair is a list of two integers.
    """
    seen = set()
    pairs = set()

    for num in nums:
        complement = target_sum - num
        if complement in seen:
            pairs.add((min(num, complement), max(num, complement)))
        seen.add(num)

    return [list(pair) for pair in pairs]

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4, 3], 6) == [[2, 4], [3, 3]]
    assert candidate([1, 5, 7, -1], 6) == [[1, 5], [7, -1]]
    assert candidate([10, 15, 3, 7], 17) == [[10, 7]]
    assert candidate([1, 2, 3, 9], 8) == []
    assert candidate([1, 2, 3, 4, 5], 10) == [[5, 5]]
    assert candidate([1, 1, 1, 1], 2) == [[1, 1]]
    assert candidate([], 5) == []
    assert candidate([5], 5) == []
    assert candidate([0, 0, 0, 0], 0) == [[0, 0]]
    assert candidate([-1, -2, -3, -4, -5], -8) == [[-3, -5], [-4, -4]]

def test_check():
    check(find_unique_pairs)
