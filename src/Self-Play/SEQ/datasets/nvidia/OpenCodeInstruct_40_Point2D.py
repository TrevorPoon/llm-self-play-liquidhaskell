from typing import *

class Point2D:
    def __init__(self, x, y):
        """
        Initializes a new Point2D object with the given x and y coordinates.

        :param x: The x-coordinate of the point.
        :param y: The y-coordinate of the point.
        """
        self.x = x
        self.y = y

    def __eq__(self, other):
        """
        Compares this Point2D object with another object for equality based on their coordinates.

        :param other: The object to compare with.
        :return: True if the other object is a Point2D and has the same coordinates, False otherwise.
        """
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        return False

### Unit tests below ###
def check(candidate):
    assert Point2D(1, 2) == Point2D(1, 2)
    assert Point2D(0, 0) == Point2D(0, 0)
    assert Point2D(-1, -1) == Point2D(-1, -1)
    assert Point2D(10, 20) != Point2D(20, 10)
    assert Point2D(5, 5) != Point2D(5, 6)
    assert Point2D(3, 4) != Point2D(4, 3)
    assert Point2D(100, 200) != Point2D(100, 201)
    assert Point2D(0, 0) != Point2D(1, 1)
    assert Point2D(1, 1) != "Point2D(1, 1)"
    assert Point2D(1, 1) != None

def test_check():
    check(Point2D())
