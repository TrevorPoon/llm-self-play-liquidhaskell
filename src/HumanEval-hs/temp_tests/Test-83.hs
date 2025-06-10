-- Task ID: HumanEval/83
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def starts_one_ends(n):
--     """
--     Given a positive integer n, return the count of the numbers of n-digit
--     positive integers that start or end with 1.
--     """
--     if n == 1: return 1
--     return 18 * (10 ** (n - 2))
-- 


-- Haskell Implementation:

-- Given a positive integer n, return the count of the numbers of n-digit
-- positive integers that start or end with 1.

starts_one_ends :: Int -> Int
starts_one_ends n =  if n == 1  then 1  else 18 * (10 ^ (n - 2))



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (starts_one_ends 1 == 1)
    check (starts_one_ends 2 == 18)
    check (starts_one_ends 3 == 180)
    check (starts_one_ends 4 == 1800)
    check (starts_one_ends 5 == 18000)
