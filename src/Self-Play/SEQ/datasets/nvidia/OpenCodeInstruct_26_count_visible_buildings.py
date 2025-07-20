from typing import *

def count_visible_buildings(heights):
    """
    Counts the number of buildings visible from a viewpoint to the left of the first building.
    
    :param heights: List of integers representing the heights of buildings.
    :return: Integer count of visible buildings.
    """
    visible_count = 0
    max_height_so_far = 0
    
    for height in heights:
        if height > max_height_so_far:
            visible_count += 1
            max_height_so_far = height
    
    return visible_count

### Unit tests below ###
def check(candidate):
    assert candidate([3, 7, 8, 3, 6, 1]) == 3
    assert candidate([1, 2, 3, 4, 5]) == 5
    assert candidate([5, 4, 3, 2, 1]) == 1
    assert candidate([10]) == 1
    assert candidate([1, 1, 1, 1, 1]) == 1
    assert candidate([1, 3, 2, 4, 3, 5]) == 4
    assert candidate([100, 200, 150, 250, 300, 275]) == 4
    assert candidate([1, 2, 3, 2, 1, 2, 3]) == 4
    assert candidate([5, 5, 5, 5, 6, 6, 7, 7, 8, 8]) == 4
    assert candidate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 10

def test_check():
    check(count_visible_buildings)
