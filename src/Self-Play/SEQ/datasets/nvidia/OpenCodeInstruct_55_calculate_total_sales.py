from typing import *

def calculate_total_sales(data):
    """
    Calculates the total sales by applying discounts to the original price and quantity.
    
    Parameters:
    data (pd.DataFrame): A DataFrame containing `price`, `quantity`, and `discount` columns.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional `total_sales` column.
    """
    data['total_sales'] = data['price'] * (1 - data['discount'] / 100) * data['quantity']
    return data

### Unit tests below ###
def check(candidate):
    assert candidate(pd.DataFrame({'price': [100], 'quantity': [1], 'discount': [0]}))['total_sales'].iloc[0] == 100
    assert candidate(pd.DataFrame({'price': [100], 'quantity': [1], 'discount': [50]}))['total_sales'].iloc[0] == 50
    assert candidate(pd.DataFrame({'price': [200], 'quantity': [2], 'discount': [25]}))['total_sales'].iloc[0] == 300
    assert candidate(pd.DataFrame({'price': [150], 'quantity': [3], 'discount': [10]}))['total_sales'].iloc[0] == 405
    assert candidate(pd.DataFrame({'price': [0], 'quantity': [10], 'discount': [0]}))['total_sales'].iloc[0] == 0
    assert candidate(pd.DataFrame({'price': [100], 'quantity': [0], 'discount': [10]}))['total_sales'].iloc[0] == 0
    assert candidate(pd.DataFrame({'price': [100], 'quantity': [1], 'discount': [100]}))['total_sales'].iloc[0] == 0
    assert candidate(pd.DataFrame({'price': [100, 200], 'quantity': [1, 2], 'discount': [0, 50]}))['total_sales'].tolist() == [100, 100]
    assert candidate(pd.DataFrame({'price': [100, 200, 150], 'quantity': [2, 3, 1], 'discount': [10, 5, 20]}))['total_sales'].tolist() == [180.0, 570.0, 120.0]
    assert candidate(pd.DataFrame({'price': [100, 200, 150], 'quantity': [2, 3, 1], 'discount': [10, 5, 20]})).shape[1] == 4

def test_check():
    check(calculate_total_sales)
