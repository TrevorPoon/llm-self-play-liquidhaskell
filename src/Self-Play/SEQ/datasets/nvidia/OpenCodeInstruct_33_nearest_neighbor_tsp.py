from typing import *

def nearest_neighbor_tsp(distance_matrix, start=0):
    """
    Solves the Traveling Salesman Problem using the Nearest Neighbor heuristic.
    
    :param distance_matrix: A 2D list representing the distances between cities.
    :param start: The index of the starting city.
    :return: A list of indices representing the order of cities to visit.
    """
    n = len(distance_matrix)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: distance_matrix[last][city])
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour

### Unit tests below ###
def check(candidate):
    assert optimize_route([(0, 0), (1, 1)]) == [(0, 0), (1, 1), (0, 0)]
    assert optimize_route([(0, 0), (0, 1), (1, 0)]) == [(0, 0), (0, 1), (1, 0), (0, 0)]
    assert optimize_route([(0, 0), (1, 1), (2, 2)]) == [(0, 0), (1, 1), (2, 2), (0, 0)]
    assert optimize_route([(0, 0), (1, 0), (0, 1), (1, 1)]) == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert optimize_route([(0, 0), (10, 0), (10, 10), (0, 10)]) == [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    assert optimize_route([(0, 0), (0, 0), (0, 0)]) == [(0, 0), (0, 0), (0, 0), (0, 0)]
    assert optimize_route([(1, 1), (2, 2), (3, 3), (4, 4)]) == [(1, 1), (2, 2), (3, 3), (4, 4), (1, 1)]
    assert optimize_route([(0, 0)]) == [(0, 0), (0, 0)]
    assert optimize_route([(0, 0), (1, 2), (3, 4), (5, 6), (7, 8)]) == [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (0, 0)]
    assert optimize_route([(1, 1), (1, 1), (2, 2), (2, 2)]) == [(1, 1), (1, 1), (2, 2), (2, 2), (1, 1)]

def test_check():
    check(nearest_neighbor_tsp)
