from typing import *

def calculate_average_grade(grades):
    """
    Calculate the average grade from a dictionary of grades.

    Parameters:
    grades (dict): A dictionary where keys are course names and values are grades.

    Returns:
    float: The average grade rounded to two decimal places.
    """
    if not grades:
        return 0.0
    total = sum(grades.values())
    average = total / len(grades)
    return round(average, 2)

### Unit tests below ###
def check(candidate):
    assert candidate({'Math': 85, 'Science': 90, 'History': 78}) == 84.33
    assert candidate({'Math': 100, 'Science': 100, 'History': 100}) == 100.00
    assert candidate({'Math': 50, 'Science': 50, 'History': 50}) == 50.00
    assert candidate({'Math': 85.5, 'Science': 90.5, 'History': 78.5}) == 84.83
    assert candidate({'Math': 95, 'Science': 80}) == 87.50
    assert candidate({}) == 0.00
    assert candidate({'Math': 100}) == 100.00
    assert candidate({'Math': 0, 'Science': 0, 'History': 0}) == 0.00
    assert candidate({'Math': 99.99, 'Science': 100.00, 'History': 99.98}) == 100.00
    assert candidate({'Math': 88.888, 'Science': 99.999, 'History': 77.777}) == 88.89

def test_check():
    check(calculate_average_grade)
