from typing import *

def max_non_overlapping_tasks(tasks):
    """
    Returns the maximum number of non-overlapping tasks that can be selected from a list of tasks.
    
    :param tasks: List of tuples, where each tuple (start, end) represents the start and end times of a task.
    :return: Integer representing the maximum number of non-overlapping tasks.
    """
    if not tasks:
        return 0

    count = 1
    last_end = tasks[0][1]

    for i in range(1, len(tasks)):
        current_start, current_end = tasks[i]
        if current_start >= last_end:
            count += 1
            last_end = current_end

    return count

### Unit tests below ###
def check(candidate):
    assert candidate([(1, 3), (2, 4), (3, 5)]) == 2
    assert candidate([(1, 2), (3, 4), (5, 6)]) == 3
    assert candidate([(1, 5), (2, 3), (4, 6)]) == 2
    assert candidate([(1, 3), (4, 6), (7, 9), (10, 12)]) == 4
    assert candidate([(1, 2)]) == 1
    assert candidate([]) == 0
    assert candidate([(1, 10), (10, 20), (20, 30)]) == 3
    assert candidate([(1, 2), (2, 3), (3, 4), (4, 5)]) == 4
    assert candidate([(1, 100), (50, 150), (100, 200)]) == 2
    assert candidate([(1, 2), (1, 3), (1, 4)]) == 1

def test_check():
    check(max_non_overlapping_tasks)
