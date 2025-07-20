from typing import *

class Calculator:
    def add(self, a, b):
        """Return the sum of a and b."""
        return a + b

    def subtract(self, a, b):
        """Return the result of subtracting b from a."""
        return a - b

    def multiply(self, a, b):
        """Return the product of a and b."""
        return a * b

    def divide(self, a, b):
        """Return the result of dividing a by b. Raises ValueError if b is zero."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def to_binary(self, num):
        """Convert num to a binary string without the '0b' prefix."""
        return bin(num)[2:]

    def to_octal(self, num):
        """Convert num to an octal string without the '0o' prefix."""
        return oct(num)[2:]

    def to_hexadecimal(self, num):
        """Convert num to a hexadecimal string without the '0x' prefix."""
        return hex(num)[2:]

### Unit tests below ###
def check(candidate):
    assert candidate.add(2, 3) == 5
    assert candidate.subtract(5, 3) == 2
    assert candidate.multiply(4, 3) == 12
    assert candidate.divide(10, 2) == 5.0
    assert candidate.to_binary(10) == "1010"
    assert candidate.to_octal(10) == "12"
    assert candidate.to_hexadecimal(10) == "a"
    assert candidate.divide(10, 0) == "Cannot divide by zero."
    assert candidate.add(-5, 5) == 0
    assert candidate.to_binary(0) == "0"

def test_check():
    check(Calculator())
