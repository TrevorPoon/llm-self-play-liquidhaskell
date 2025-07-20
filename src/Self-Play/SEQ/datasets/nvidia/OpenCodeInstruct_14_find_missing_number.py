from typing import *

def find_missing_number(nums):
    """
    Finds the missing number in the range [1, n] from the given list of n-1 distinct integers.

    :param nums: List[int] - A list of n-1 distinct integers in the range [1, n]
    :return: int - The missing number in the range [1, n]
    """
    n = len(nums) + 1
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 4, 5, 6]) == 3
    assert candidate([1, 3, 4, 5]) == 2
    assert candidate([2, 3, 4, 5, 6]) == 1
    assert candidate([1, 2, 3, 4, 5, 7]) == 6
    assert candidate([1]) == 2
    assert candidate([2]) == 1
    assert candidate([]) == 1
    assert candidate([1, 2, 3, 5]) == 4
    assert candidate([1, 2, 3, 4, 6]) == 5
    assert candidate([5, 2, 3, 1]) == 4

def test_check():
    check(find_missing_number)
