from typing import *

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

### Unit tests below ###
def check(candidate):
    assert BinaryTree().find_minimum() == None
    assert BinaryTree().in_order_traversal() == []
    bt = BinaryTree(); bt.insert(10); assert bt.find_minimum() == 10
    bt = BinaryTree(); bt.insert(10); bt.insert(5); assert bt.find_minimum() == 5
    bt = BinaryTree(); bt.insert(10); bt.insert(5); bt.insert(15); assert bt.in_order_traversal() == [5, 10, 15]
    bt = BinaryTree(); bt.insert(10); bt.insert(5); bt.insert(15); bt.insert(3); assert bt.in_order_traversal() == [3, 5, 10, 15]
    bt = BinaryTree(); bt.insert(10); bt.insert(5); bt.insert(15); bt.insert(3); bt.insert(7); assert bt.in_order_traversal() == [3, 5, 7, 10, 15]
    bt = BinaryTree(); bt.insert(10); bt.insert(5); bt.insert(15); bt.insert(3); bt.insert(7); bt.insert(12); assert bt.in_order_traversal() == [3, 5, 7, 10, 12, 15]
    bt = BinaryTree(); bt.insert(10); bt.insert(5); bt.insert(15); bt.insert(3); bt.insert(7); bt.insert(12); bt.insert(18); assert bt.in_order_traversal() == [3, 5, 7, 10, 12, 15, 18]
    bt = BinaryTree(); bt.insert(10); bt.insert(5); bt.insert(15); bt.insert(3); bt.insert(7); bt.insert(12); bt.insert(18); bt.insert(1); assert bt.in_order_traversal() == [1, 3, 5, 7, 10, 12, 15, 18]

def test_check():
    check(TreeNode())
