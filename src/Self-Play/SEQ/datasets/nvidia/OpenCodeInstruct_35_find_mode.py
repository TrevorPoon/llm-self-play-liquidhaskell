from typing import *

def find_mode(numbers):
    """
    Finds the mode(s) of a list of numbers.
    
    Parameters:
    numbers (list): A list of integers.
    
    Returns:
    int or list: The mode as an integer if there's only one mode, 
                 or a list of modes if there are multiple.
    """
    from collections import Counter
    
    counts = Counter(numbers)
    max_freq = max(counts.values())
    modes = [num for num, freq in counts.items() if freq == max_freq]
    
    return modes[0] if len(modes) == 1 else modes

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 2, 3, 3, 4]) == [2, 3]
    assert candidate([4, 4, 1, 2, 2, 3, 3]) == [2, 3, 4]
    assert candidate([7, 7, 7, 1, 2, 2, 3]) == 7
    assert candidate([1, 1, 2, 3, 4, 5, 5]) == [1, 5]
    assert candidate([10]) == 10
    assert candidate([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert candidate([2, 2, 3, 3, 3, 4, 4, 4]) == [3, 4]
    assert candidate([]) == []
    assert candidate([5, 5, 5, 5, 5]) == 5
    assert candidate([1, 2, 3, 3, 2, 1]) == [1, 2, 3]

def test_check():
    check(find_mode)
