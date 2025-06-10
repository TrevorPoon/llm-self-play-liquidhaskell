-- Task ID: HumanEval/60
-- Assigned To: Author C

-- Python Implementation:

-- 
-- 
-- def sum_to_n(n: int):
--     """sum_to_n is a function that sums numbers from 1 to n.
--     >>> sum_to_n(30)
--     465
--     >>> sum_to_n(100)
--     5050
--     >>> sum_to_n(5)
--     15
--     >>> sum_to_n(10)
--     55
--     >>> sum_to_n(1)
--     1
--     """
--     return sum(range(n + 1))
-- 


-- Haskell Implementation:

-- sum_to_n is a function that sums numbers from 1 to n.
-- >>> sum_to_n 30
-- 465
-- >>> sum_to_n 100
-- 5050
-- >>> sum_to_n 5
-- 15
-- >>> sum_to_n 10
-- 55
-- >>> sum_to_n 1
-- 1
sum_to_n :: Int -> Int
sum_to_n n =  sum [1..n]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (sum_to_n 1 == 1)
    check (sum_to_n 6 == 21)
    check (sum_to_n 11 == 66)
    check (sum_to_n 30 == 465)
    check (sum_to_n 100 == 5050)