from typing import *

def generate_random_color():
    """
    Generates a random color in the RGB color model.

    Returns:
        tuple: A tuple of three integers representing the RGB values, each ranging from 0 to 255.
    """
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return (red, green, blue)

### Unit tests below ###
def check(candidate):
    assert isinstance(candidate(), tuple)
    assert len(candidate()) == 3
    assert all(isinstance(x, int) for x in candidate())
    assert all(0 <= x <= 255 for x in candidate())
    assert candidate() != candidate()
    assert candidate() != (0, 0, 0) or candidate() != (255, 255, 255)
    assert candidate() != (128, 128, 128)
    assert candidate() != (255, 0, 0) or candidate() != (0, 255, 0) or candidate() != (0, 0, 255)
    assert candidate() != (255, 255, 0) or candidate() != (0, 255, 255) or candidate() != (255, 0, 255)
    assert candidate() != (192, 192, 192)

def test_check():
    check(generate_random_color)
