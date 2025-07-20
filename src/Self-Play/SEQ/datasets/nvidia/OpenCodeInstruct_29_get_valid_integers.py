from typing import *

def get_valid_integers():
    """
    Prompts the user to enter a series of integers separated by spaces.
    Repeats the prompt until valid input is provided.
    
    Returns:
        list of int: A list of integers entered by the user.
    """
    while True:
        user_input = input("Please enter a series of integers separated by spaces: ").strip()
        numbers = user_input.split()
        if all(num.isdigit() or (num.startswith('-') and num[1:].isdigit()) for num in numbers):
            return list(map(int, numbers))
        else:
            print("Invalid input. Please make sure to enter only integers separated by spaces.")

### Unit tests below ###
def check(candidate):
    assert calculate_sum([1, 2, 3, 4, 5]) == 15
    assert calculate_sum([-1, -2, -3, -4, -5]) == -15
    assert calculate_sum([0, 0, 0, 0, 0]) == 0
    assert calculate_sum([10, -10, 20, -20, 30]) == 30
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([-1, -2, -3, -4, -5]) == -3.0
    assert calculate_average([0, 0, 0, 0, 0]) == 0.0
    assert calculate_average([10, -10, 20, -20, 30]) == 6.0
    assert calculate_product([1, 2, 3, 4, 5]) == 120
    assert calculate_product([-1, -2, -3, -4, -5]) == -120

def test_check():
    check(get_valid_integers)
