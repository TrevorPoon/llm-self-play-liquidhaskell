from typing import *

class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

### Unit tests below ###
def check(candidate):
    assert flipBinaryTree(None) is None
    assert flipBinaryTree(TreeNode(1)).value == 1 and flipBinaryTree(TreeNode(1)).left is None and flipBinaryTree(TreeNode(1)).right is None
    root = TreeNode(1, TreeNode(2), TreeNode(3))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, TreeNode(6), TreeNode(7)))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.left.left.value == 7 and flipped_root.left.right.value == 6 and flipped_root.right.left.value == 5 and flipped_root.right.right.value == 4
    root = TreeNode(1, TreeNode(2, TreeNode(4, TreeNode(8), TreeNode(9)), TreeNode(5)), TreeNode(3, TreeNode(6), TreeNode(7, TreeNode(10), TreeNode(11))))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.left.left.value == 7 and flipped_root.left.right.value == 6 and flipped_root.left.left.left.value == 11 and flipped_root.left.left.right.value == 10 and flipped_root.right.left.value == 5 and flipped_root.right.right.value == 4 and flipped_root.right.right.left.value == 9 and flipped_root.right.right.right.value == 8
    root = TreeNode(1, TreeNode(2, TreeNode(4, TreeNode(8), TreeNode(9)), TreeNode(5, TreeNode(10), TreeNode(11))), TreeNode(3, TreeNode(6, TreeNode(12), TreeNode(13)), TreeNode(7)))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.left.left.value == 7 and flipped_root.left.right.value == 6 and flipped_root.left.right.left.value == 13 and flipped_root.left.right.right.value == 12 and flipped_root.right.left.value == 5 and flipped_root.right.right.value == 4 and flipped_root.right.left.left.value == 11 and flipped_root.right.left.right.value == 10 and flipped_root.right.right.left.value == 9 and flipped_root.right.right.right.value == 8
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.right.left.value == 5 and flipped_root.right.right.value == 4
    root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(6), TreeNode(7)))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.left.left.value == 7 and flipped_root.left.right.value == 6
    root = TreeNode(1, TreeNode(2, TreeNode(4, TreeNode(8))), TreeNode(3, None, TreeNode(7, None, TreeNode(11))))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.left.right.value == 7 and flipped_root.left.right.right.value == 11 and flipped_root.right.left.value == 4 and flipped_root.right.left.left.value == 8
    root = TreeNode(1, TreeNode(2, None, TreeNode(5, TreeNode(10), TreeNode(11))), TreeNode(3, TreeNode(6, TreeNode(12), TreeNode(13)), None))
    flipped_root = flipBinaryTree(root)
    assert flipped_root.value == 1 and flipped_root.left.value == 3 and flipped_root.right.value == 2 and flipped_root.left.left.value == 6 and flipped_root.left.left.left.value == 13 and flipped_root.left.left.right.value == 12 and flipped_root.right.right.value == 5 and flipped_root.right.right.left.value == 10 and flipped_root.right.right.right.value == 11

def test_check():
    check(TreeNode())
