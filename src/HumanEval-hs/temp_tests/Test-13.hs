-- Task ID: HumanEval/13
-- Assigned To: Author A

-- Python Implementation:

-- 
-- 
-- def greatest_common_divisor(a: int, b: int) -> int:
--     """ Return a greatest common divisor of two integers a and b
--     >>> greatest_common_divisor(3, 5)
--     1
--     >>> greatest_common_divisor(25, 15)
--     5
--     """
--     while b:
--         a, b = b, a % b
--     return a
-- 


-- Haskell Implementation:

-- Return a greatest common divisor of two integers a and b
-- >>> greatest_common_divisor 3 5
-- 1
-- >>> greatest_common_divisor 25 15
-- 5
greatest_common_divisor :: Int -> Int -> Int
greatest_common_divisor a b =  gcd  a b



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (greatest_common_divisor 3 7   == 1)
    check (greatest_common_divisor 10 15 == 5)
    check (greatest_common_divisor 49 14 == 7)
    check (greatest_common_divisor 144 60 == 12)
