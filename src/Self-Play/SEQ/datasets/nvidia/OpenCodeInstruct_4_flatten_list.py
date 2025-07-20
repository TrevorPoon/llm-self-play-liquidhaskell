from typing import *

def flatten_list(nested_list):
    """
    Recursively flattens a nested list of integers.

    :param nested_list: List of integers and/or nested lists of integers.
    :return: A flat list containing all integers from the nested list.
    """
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list

### Unit tests below ###
def check(candidate):
    assert candidate([1, [2, [3, 4], 5], 6, [7, 8]]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert candidate([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    assert candidate([[1], [2], [3], [4], [5]]) == [1, 2, 3, 4, 5]
    assert candidate([[[[1]]]]) == [1]
    assert candidate([]) == []
    assert candidate([[], [], []]) == []
    assert candidate([1, [2, [3, [4, [5]]]]]) == [1, 2, 3, 4, 5]
    assert candidate([1, [2, 3], [4, [5, 6]], 7]) == [1, 2, 3, 4, 5, 6, 7]
    assert candidate([1, [2, [3, [4, [5, [6, [7, [8, [9]]]]]]]]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert candidate([1, [2, [3, [4, [5, [6, [7, [8, [9, []]]]]]]]]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

def test_check():
    check(flatten_list)
