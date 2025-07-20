from typing import *

def calculate_discounted_price(original_price, discount_percentage):
    """
    Calculate the discounted price of a product given its original price and discount percentage.
    
    Parameters:
    original_price (float or int): The original price of the product, must be a positive number.
    discount_percentage (float or int): The discount percentage to be applied, must be between 0 and 100 inclusive.
    
    Returns:
    float: The discounted price of the product.
    
    Raises:
    ValueError: If original_price is not positive or discount_percentage is not between 0 and 100.
    """
    if not (isinstance(original_price, (int, float)) and original_price > 0):
        raise ValueError("Original price must be a positive number.")
    if not (isinstance(discount_percentage, (int, float)) and 0 <= discount_percentage <= 100):
        raise ValueError("Discount percentage must be a number between 0 and 100.")
    
    discount_amount = original_price * (discount_percentage / 100)
    discounted_price = original_price - discount_amount
    return discounted_price

### Unit tests below ###
def check(candidate):
    assert candidate(100, 20) == 80
    assert candidate(200, 50) == 100
    assert candidate(150, 0) == 150
    assert candidate(150, 100) == 0
    assert candidate(99.99, 10) == 89.991
    assert candidate(50, 5) == 47.5
    assert candidate(1000, 10) == 900
    assert candidate(100, 0.5) == 99.5
    assert candidate(100, 100.0) == 0
    assert candidate(100, 0.0) == 100

def test_check():
    check(calculate_discounted_price)
