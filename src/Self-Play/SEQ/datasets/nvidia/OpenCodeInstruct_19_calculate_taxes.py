from typing import *

def calculate_taxes(incomes):
    """
    Calculate the taxes for a list of income amounts based on progressive tax brackets.

    :param incomes: List of integers representing income amounts.
    :return: List of floats representing the tax amounts for each income.
    """
    taxes = []
    for income in incomes:
        if income <= 10000:
            tax = income * 0.10
        elif income <= 20000:
            tax = 10000 * 0.10 + (income - 10000) * 0.20
        else:
            tax = 10000 * 0.10 + 10000 * 0.20 + (income - 20000) * 0.30
        taxes.append(tax)
    return taxes

### Unit tests below ###
def check(candidate):
    assert candidate([0]) == [0.0]
    assert candidate([5000]) == [500.0]
    assert candidate([10000]) == [1000.0]
    assert candidate([10001]) == [2000.2]
    assert candidate([20000]) == [3000.0]
    assert candidate([20001]) == [3000.3]
    assert candidate([5000, 15000, 25000]) == [500.0, 2000.0, 4500.0]
    assert candidate([10000, 20000, 30000]) == [1000.0, 3000.0, 6000.0]
    assert candidate([9999, 10000, 10001]) == [999.9, 1000.0, 2000.2]
    assert candidate([25000, 30000, 35000]) == [4500.0, 5500.0, 6500.0]

def test_check():
    check(calculate_taxes)
