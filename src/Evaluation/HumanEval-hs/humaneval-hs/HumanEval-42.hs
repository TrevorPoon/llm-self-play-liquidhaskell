-- Task ID: HumanEval/42
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def incr_list(l: list):
--     """Return list with elements incremented by 1.
--     >>> incr_list([1, 2, 3])
--     [2, 3, 4]
--     >>> incr_list([5, 3, 5, 2, 3, 3, 9, 0, 123])
--     [6, 4, 6, 3, 4, 4, 10, 1, 124]
--     """
--     return [(e + 1) for e in l]
-- 


-- Haskell Implementation:

-- Return list with elements incremented by 1.
-- >>> incr_list [1,2,3]
-- [2,3,4]
-- >>> incr_list [5,3,5,2,3,3,9,0,123]
-- [6,4,6,3,4,4,10,1,124]
incr_list :: [Int] -> [Int]
incr_list =  map (+1)
