from typing import *

def cube_elements(input_list):
    """
    Returns a new list where each element is the cube of the corresponding element in the input list.

    :param input_list: List of integers
    :return: List of integers where each element is the cube of the corresponding element in the input list
    """
    return [x**3 for x in input_list]

### Unit tests below ###
def check(candidate):
    assert candidate([1, 2, 3, 4]) == [1, 8, 27, 64]
    assert candidate([0, 0, 0]) == [0, 0, 0]
    assert candidate([-1, -2, -3]) == [-1, -8, -27]
    assert candidate([5, 10, 15]) == [125, 1000, 3375]
    assert candidate([2, 4, 6, 8]) == [8, 64, 216, 512]
    assert candidate([]) == []
    assert candidate([10]) == [1000]
    assert candidate([-10, 0, 10]) == [-1000, 0, 1000]
    assert candidate([7, -7, 7]) == [343, -343, 343]
    assert candidate([1, -1, 1, -1]) == [1, -1, 1, -1]

def test_check():
    check(cube_elements)
