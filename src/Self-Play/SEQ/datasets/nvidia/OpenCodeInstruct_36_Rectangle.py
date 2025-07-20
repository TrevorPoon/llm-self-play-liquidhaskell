from typing import *

class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        """Getter for width."""
        return self._width

    @width.setter
    def width(self, value):
        """Setter for width with validation."""
        if value < 0:
            raise ValueError("Width must be non-negative")
        self._width = value

    @property
    def height(self):
        """Getter for height."""
        return self._height

    @height.setter
    def height(self, value):
        """Setter for height with validation."""
        if value < 0:
            raise ValueError("Height must be non-negative")
        self._height = value

    def area(self):
        """Calculates and returns the area of the rectangle."""
        return self._width * self._height

    def perimeter(self):
        """Calculates and returns the perimeter of the rectangle."""
        return 2 * (self._width + self._height)

    def resize(self, factor):
        """Resizes the rectangle by a given factor."""
        if factor < 0:
            raise ValueError("Resize factor must be non-negative")
        self._width *= factor
        self._height *= factor

    def display_info(self):
        """Prints the width, height, area, and perimeter of the rectangle."""
        print(f"Width: {self._width}, Height: {self._height}, "
              f"Area: {self.area()}, Perimeter: {self.perimeter()}")

### Unit tests below ###
def check(candidate):
    assert Rectangle(3, 4).area() == 12
    assert Rectangle(5, 5).perimeter() == 20
    rect = Rectangle(2, 3)
    rect.resize(2)
    assert rect.width == 4 and rect.height == 6
    rect = Rectangle(10, 2)
    rect.resize(0.5)
    assert rect.width == 5 and rect.height == 1
    rect = Rectangle(7, 3)
    rect.width = 14
    assert rect.width == 14
    rect = Rectangle(7, 3)
    rect.height = 6
    assert rect.height == 6
    try:
    rect = Rectangle(-1, 5)
    except ValueError as e:
    assert str(e) == "Width must be non-negative"
    try:
    rect = Rectangle(5, -1)
    except ValueError as e:
    assert str(e) == "Height must be non-negative"
    try:
    rect = Rectangle(5, 5)
    rect.resize(-1)
    except ValueError as e:
    assert str(e) == "Resize factor must be non-negative"
    rect = Rectangle(8, 4)
    assert rect.area() == 32 and rect.perimeter() == 24

def test_check():
    check(Rectangle())
