from typing import *

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

### Unit tests below ###
def check(candidate):
    assert find_closest_pair_of_points([(0, 0), (1, 1)]) == ((0, 0), (1, 1))
    assert find_closest_pair_of_points([(1, 2), (3, 4), (5, 6), (7, 8), (1, 3)]) == ((1, 2), (1, 3))
    assert find_closest_pair_of_points([(0, 0), (10, 10), (5, 5)]) == ((0, 0), (5, 5))
    assert find_closest_pair_of_points([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]) == ((1, 1), (2, 2))
    assert find_closest_pair_of_points([(0, 0), (0, 1), (1, 0), (1, 1)]) == ((0, 0), (0, 1))
    assert find_closest_pair_of_points([(10, 10), (20, 20), (15, 15)]) == ((10, 10), (15, 15))
    assert find_closest_pair_of_points([(1, 1), (1, 2), (2, 1), (2, 2)]) == ((1, 1), (1, 2))
    assert find_closest_pair_of_points([(0, 0), (1, 0), (0, 1), (1, 1)]) == ((0, 0), (1, 0))
    assert find_closest_pair_of_points([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]) == ((1, 1), (2, 2))
    assert find_closest_pair_of_points([(1, 1), (1, 1), (2, 2)]) == ((1, 1), (1, 1))

def test_check():
    check(euclidean_distance)
