from typing import *

def sort_products_by_price(products):
    """
    Sorts a list of product tuples by price in ascending order.
    If two products have the same price, their order remains unchanged.

    :param products: List of tuples, where each tuple contains a product name and its price.
    :return: List of tuples sorted by price in ascending order.
    """
    return sorted(products, key=lambda product: product[1])

### Unit tests below ###
def check(candidate):
    assert candidate([("Apple", 1.20), ("Banana", 0.99), ("Cherry", 1.20), ("Date", 2.50)]) == [("Banana", 0.99), ("Apple", 1.20), ("Cherry", 1.20), ("Date", 2.50)]
    assert candidate([("Grape", 2.00), ("Orange", 1.50), ("Peach", 1.50), ("Plum", 3.00)]) == [("Orange", 1.50), ("Peach", 1.50), ("Grape", 2.00), ("Plum", 3.00)]
    assert candidate([("Kiwi", 1.00), ("Lemon", 1.00), ("Mango", 1.00)]) == [("Kiwi", 1.00), ("Lemon", 1.00), ("Mango", 1.00)]
    assert candidate([("Nectarine", 3.50), ("Papaya", 2.75), ("Quince", 2.75), ("Raspberry", 4.00)]) == [("Papaya", 2.75), ("Quince", 2.75), ("Nectarine", 3.50), ("Raspberry", 4.00)]
    assert candidate([("Strawberry", 5.00)]) == [("Strawberry", 5.00)]
    assert candidate([]) == []
    assert candidate([("Tomato", 0.75), ("Ugli", 0.75), ("Vanilla", 0.75), ("Watermelon", 0.75)]) == [("Tomato", 0.75), ("Ugli", 0.75), ("Vanilla", 0.75), ("Watermelon", 0.75)]
    assert candidate([("Xigua", 1.25), ("Yam", 1.25), ("Zucchini", 1.25), ("Avocado", 1.25)]) == [("Xigua", 1.25), ("Yam", 1.25), ("Zucchini", 1.25), ("Avocado", 1.25)]
    assert candidate([("Blueberry", 2.25), ("Cantaloupe", 2.25), ("Dragonfruit", 2.25), ("Elderberry", 2.25)]) == [("Blueberry", 2.25), ("Cantaloupe", 2.25), ("Dragonfruit", 2.25), ("Elderberry", 2.25)]
    assert candidate([("Fig", 3.75), ("Grapefruit", 3.75), ("Honeydew", 3.75), ("Icaco", 3.75)]) == [("Fig", 3.75), ("Grapefruit", 3.75), ("Honeydew", 3.75), ("Icaco", 3.75)]

def test_check():
    check(sort_products_by_price)
