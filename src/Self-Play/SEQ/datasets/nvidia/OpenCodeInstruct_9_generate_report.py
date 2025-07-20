from typing import *

def generate_report(sales):
    """
    Generates a report summarizing the total quantity sold and total revenue from a list of sales data.

    Parameters:
    sales (list of dict): A list where each dictionary contains 'product', 'quantity', and 'price' keys.

    Returns:
    str: A string summarizing the total quantity sold and total revenue.
    """
    total_quantity = 0
    total_revenue = 0

    for sale in sales:
        total_quantity += sale['quantity']
        total_revenue += sale['quantity'] * sale['price']

    return f"Total Quantity Sold: {total_quantity}, Total Revenue: ${total_revenue}"

### Unit tests below ###
def check(candidate):
    assert candidate([{"product": "Laptop", "quantity": 2, "price": 800}, {"product": "Smartphone", "quantity": 5, "price": 300}]) == "Total Quantity Sold: 7, Total Revenue: $5100"
    assert candidate([{"product": "Book", "quantity": 10, "price": 15}]) == "Total Quantity Sold: 10, Total Revenue: $150"
    assert candidate([]) == "Total Quantity Sold: 0, Total Revenue: $0"
    assert candidate([{"product": "Pen", "quantity": 100, "price": 0.5}]) == "Total Quantity Sold: 100, Total Revenue: $50.0"
    assert candidate([{"product": "Tablet", "quantity": 3, "price": 250}, {"product": "Monitor", "quantity": 2, "price": 150}]) == "Total Quantity Sold: 5, Total Revenue: $1150"
    assert candidate([{"product": "Coffee", "quantity": 50, "price": 2}, {"product": "Tea", "quantity": 30, "price": 1.5}]) == "Total Quantity Sold: 80, Total Revenue: $145.0"
    assert candidate([{"product": "Desk", "quantity": 1, "price": 1000}, {"product": "Chair", "quantity": 4, "price": 200}]) == "Total Quantity Sold: 5, Total Revenue: $2200"
    assert candidate([{"product": "Bike", "quantity": 2, "price": 500}, {"product": "Car", "quantity": 1, "price": 15000}]) == "Total Quantity Sold: 3, Total Revenue: $15500"
    assert candidate([{"product": "Shoes", "quantity": 10, "price": 50}, {"product": "Hat", "quantity": 20, "price": 25}]) == "Total Quantity Sold: 30, Total Revenue: $1000"
    assert candidate([{"product": "Guitar", "quantity": 0, "price": 300}, {"product": "Drums", "quantity": 1, "price": 500}]) == "Total Quantity Sold: 1, Total Revenue: $500"

def test_check():
    check(generate_report)
