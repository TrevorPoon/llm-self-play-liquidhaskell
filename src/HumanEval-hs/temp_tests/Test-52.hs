-- Task ID: HumanEval/52
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def below_threshold(l: list, t: int):
--     """Return True if all numbers in the list l are below threshold t.
--     >>> below_threshold([1, 2, 4, 10], 100)
--     True
--     >>> below_threshold([1, 20, 4, 10], 5)
--     False
--     """
--     for e in l:
--         if e >= t:
--             return False
--     return True
-- 


-- Haskell Implementation:

-- Return True if all numbers in the list l are below threshold t.
-- >>> below_threshold [1,2,4,10] 100
-- True
-- >>> below_threshold [1,20,4,10] 5
-- False
below_threshold :: [Int] -> Int -> Bool
below_threshold numbers threshold =  all (< threshold) numbers



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (below_threshold [1,2,4,10] 100)
    check (not $ below_threshold [1,20,4,10] 5)
    check (below_threshold [1,20,4,10] 21)
    check (below_threshold [1,8,4,10] 11)
    check (not $ below_threshold [1,8,4,10] 10)