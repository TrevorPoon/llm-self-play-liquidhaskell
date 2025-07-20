from typing import *

def find_unique_pairs(lst, target):
    """
    Finds all unique pairs of numbers in the list that add up to the target sum.
    
    :param lst: List of integers.
    :param target: Integer representing the target sum.
    :return: List of unique pairs (tuples) that add up to the target.
    """
    seen = set()
    unique_pairs = set()
    
    for number in lst:
        complement = target - number
        if complement in seen:
            unique_pairs.add((min(number, complement), max(number, complement)))
        seen.add(number)
    
    return list(unique_pairs)

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4, 5], 5) == [(1, 4), (2, 3)]
    assert candidate([1, 2, 3, 4, 5], 10) == []
    assert candidate([1, 1, 2, 45, 46, 46], 47) == [(1, 46)]
    assert candidate([0, 0, 0, 0], 0) == [(0, 0)]
    assert candidate([-1, 0, 1, 2, -1, -4], 0) == [(-1, 1)]
    assert candidate([10, 12, 10, 15, -1, 7, 6, 5, 4, 2, 1, 1, 1], 11) == [(10, 1), (5, 6), (4, 7)]
    assert candidate([], 5) == []
    assert candidate([5], 5) == []
    assert candidate([3, 3, 3, 3, 3], 6) == [(3, 3)]
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9], 10) == [(1, 9), (2, 8), (3, 7), (4, 6)]

def test_check():
    check(find_unique_pairs)
