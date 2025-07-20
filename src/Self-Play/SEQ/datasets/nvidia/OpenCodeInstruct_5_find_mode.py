from typing import *

def find_mode(numbers):
    """
    Returns the mode of the list of integers. If there is a tie, returns the smallest element among them.
    Returns None if the list is empty.
    
    :param numbers: List of integers
    :return: Integer or None
    """
    if not numbers:
        return None
    
    frequency = {}
    for num in numbers:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    
    max_freq = max(frequency.values())
    modes = [num for num, freq in frequency.items() if freq == max_freq]
    
    return min(modes)

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]) == 4
    assert candidate([1, 1, 2, 2, 3, 3]) == 1
    assert candidate([7, 7, 7, 8, 8, 9]) == 7
    assert candidate([10, 20, 20, 10, 30]) == 10
    assert candidate([5, 5, 5, 5, 5]) == 5
    assert candidate([]) == None
    assert candidate([1]) == 1
    assert candidate([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]) == 5
    assert candidate([100, 200, 300, 100, 200, 300]) == 100
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 1

def test_check():
    check(find_mode)
