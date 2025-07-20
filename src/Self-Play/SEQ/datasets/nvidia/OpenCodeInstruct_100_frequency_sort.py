from typing import *

def frequency_sort(nums):
    """
    Returns a list of frequencies of integers in the input list, sorted in descending order of frequency.
    Integers with the same frequency are sorted in ascending order.
    
    :param nums: List of integers
    :return: List of frequencies
    """
    if not nums:
        return []
    freq_counter = Counter(nums)
    sorted_items = sorted(freq_counter.items(), key=lambda x: (-x[1], x[0]))
    result = [freq for num, freq in sorted_items]
    return result

### Unit tests below ###
def check(candidate):
    assert candidate([4, 6, 2, 6, 7, 4, 4]) == [3, 2, 1]
    assert candidate([1, 1, 2, 2, 2, 3]) == [3, 2, 1]
    assert candidate([5, 3, 3, 2, 2, 2, 1]) == [3, 2, 1]
    assert candidate([7, 7, 7, 7, 7]) == [5]
    assert candidate([]) == []
    assert candidate([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]) == [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert candidate([3, 1, 2, 2, 4, 4, 4, 5, 5, 5, 5]) == [4, 3, 2, 1]
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert candidate([1, 1, 1, 2, 2, 3, 3, 3, 3]) == [4, 2, 2]

def test_check():
    check(frequency_sort)
