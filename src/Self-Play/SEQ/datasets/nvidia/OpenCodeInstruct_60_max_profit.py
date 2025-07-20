from typing import *

def max_profit(prices):
    """
    Calculate the maximum profit from a single buy and sell transaction.

    :param prices: List[int] - A list of integers representing the stock prices.
    :return: int - The maximum profit that can be achieved.
    """
    if not prices:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices:
        if price < min_price:
            min_price = price
        else:
            potential_profit = price - min_price
            if potential_profit > max_profit:
                max_profit = potential_profit
                
    return max_profit

### Unit tests below ###
def check(candidate):
    assert candidate([7, 1, 5, 3, 6, 4]) == 5
    assert candidate([7, 6, 4, 3, 1]) == 0
    assert candidate([1, 2, 3, 4, 5]) == 4
    assert candidate([3, 3, 5, 0, 0, 3, 1, 4]) == 4
    assert candidate([1, 2]) == 1
    assert candidate([1]) == 0
    assert candidate([]) == 0
    assert candidate([10, 7, 5, 8, 11, 9]) == 6
    assert candidate([1, 2, 4, 2, 5, 7, 2, 4, 9, 0]) == 8
    assert candidate([1, 3, 2, 8, 4, 9]) == 8

def test_check():
    check(max_profit)
