-- Task ID: HumanEval/3
-- Assigned To: Author A

-- Python Implementation:

-- from typing import List
-- 
-- 
-- def below_zero(operations: List[int]) -> bool:
--     """ You're given a list of deposit and withdrawal operations on a bank account that starts with
--     zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
--     at that point function should return True. Otherwise it should return False.
--     >>> below_zero([1, 2, 3])
--     False
--     >>> below_zero([1, 2, -4, 5])
--     True
--     """
--     balance = 0
-- 
--     for op in operations:
--         balance += op
--         if balance < 0:
--             return True
-- 
--     return False
-- 


-- Haskell Implementation:

-- You're given a list of deposit and withdrawal operations on a bank account that starts with
-- zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
-- at that point function should return True. Otherwise it should return False.
-- >>> below_zero [1, 2, 3]
-- False
-- >>> below_zero [1, 2, -4, 5]
-- True
below_zero :: [Int] -> Bool
below_zero operations =  any (< 0) $  scanl (+) 0 operations
