from typing import *

def remove_consecutive_duplicates(nums):
    """
    Removes consecutive duplicate elements from the input list.

    :param nums: List of integers
    :return: List of integers with consecutive duplicates removed
    """
    if not nums:
        return []
    
    result = [nums[0]]
    
    for num in nums[1:]:
        if num != result[-1]:
            result.append(num)
    
    return result

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 2, 3, 3, 3, 4]) == [1, 2, 3, 4]
    assert candidate([1, 1, 1, 1, 1]) == [1]
    assert candidate([]) == []
    assert candidate([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert candidate([5, 5, 4, 4, 3, 3, 2, 2, 1, 1]) == [5, 4, 3, 2, 1]
    assert candidate([1]) == [1]
    assert candidate([1, 2, 1, 2, 1]) == [1, 2, 1, 2, 1]
    assert candidate([1, 1, 2, 2, 3, 3, 2, 2, 1, 1]) == [1, 2, 3, 2, 1]
    assert candidate([1, 2, 3, 2, 3, 4, 5, 4, 5]) == [1, 2, 3, 2, 3, 4, 5, 4, 5]
    assert candidate([1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]) == [1, 2, 3, 4]

def test_check():
    check(remove_consecutive_duplicates)
