from typing import *

def convert_to_celsius(temperatures):
    """
    Convert a list of temperatures from Fahrenheit to Celsius.

    Parameters:
    temperatures (list of float): A list of temperatures in Fahrenheit.

    Returns:
    list of float: A list of temperatures converted to Celsius.
    """
    return [(f - 32) * 5/9 for f in temperatures]

### Unit tests below ###
def check(candidate):
    assert candidate([32]) == [0.0]
    assert candidate([212]) == [100.0]
    assert candidate([98.6]) == [37.0]
    assert candidate([77]) == [25.0]
    assert candidate([0]) == [-17.77777777777778]
    assert candidate([-40]) == [-40.0]
    assert candidate([100]) == [37.77777777777778]
    assert candidate([32, 212, 98.6, 77]) == [0.0, 100.0, 37.0, 25.0]
    assert candidate([]) == []
    assert candidate([68, 86, 104]) == [20.0, 30.0, 40.0]

def test_check():
    check(convert_to_celsius)
