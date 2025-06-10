-- Task ID: HumanEval/24
-- Assigned To: Author A

-- Python Implementation:

-- 
-- 
-- def largest_divisor(n: int) -> int:
--     """ For a given number n, find the largest number that divides n evenly, smaller than n
--     >>> largest_divisor(15)
--     5
--     """
--     for i in reversed(range(n)):
--         if n % i == 0:
--             return i
-- 


-- Haskell Implementation:

-- For a given number n, find the largest number that divides n evenly, smaller than n
-- >>> largest_divisor 15
-- 5
largest_divisor :: Int -> Int
largest_divisor n =  maximum  [x | x <-  [1..n-1],  n `mod` x ==  0]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (largest_divisor 3 == 1)
    check (largest_divisor 7 == 1)
    check (largest_divisor 10 == 5)
    check (largest_divisor 100 == 50)
    check (largest_divisor 49 == 7)
