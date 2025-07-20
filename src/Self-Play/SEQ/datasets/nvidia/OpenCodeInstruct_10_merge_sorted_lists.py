from typing import *

def merge_sorted_lists(list1, list2):
    """
    Merges two sorted lists into a single sorted list without using built-in sorting functions.
    
    Parameters:
    list1 (list): The first sorted list of integers.
    list2 (list): The second sorted list of integers.
    
    Returns:
    list: A merged sorted list containing all elements from list1 and list2.
    """
    i, j = 0, 0
    merged_list = []
    
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1
    
    while i < len(list1):
        merged_list.append(list1[i])
        i += 1
    
    while j < len(list2):
        merged_list.append(list2[j])
        j += 1
    
    return merged_list

### Unit tests below ###
def check(candidate):
    assert candidate([1, 3, 5, 7], [2, 4, 6, 8]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert candidate([], []) == []
    assert candidate([1, 2, 3], []) == [1, 2, 3]
    assert candidate([], [1, 2, 3]) == [1, 2, 3]
    assert candidate([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert candidate([1, 1, 1], [1, 1, 1]) == [1, 1, 1, 1, 1, 1]
    assert candidate([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]
    assert candidate([10, 20, 30], [5, 15, 25]) == [5, 10, 15, 20, 25, 30]
    assert candidate([-1, 0, 1], [-2, -1, 0]) == [-2, -1, -1, 0, 0, 1]
    assert candidate([1], [2]) == [1, 2]

def test_check():
    check(merge_sorted_lists)
