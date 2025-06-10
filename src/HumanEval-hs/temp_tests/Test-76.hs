-- Task ID: HumanEval/76
-- Assigned To: Author E

-- Python Implementation:

-- 
-- def is_simple_power(x, n):
--     """Your task is to write a function that returns true if a number x is a simple
--     power of n and false in other cases.
--     x is a simple power of n if n**int=x
--     For example:
--     is_simple_power(1, 4) => true
--     is_simple_power(2, 2) => true
--     is_simple_power(8, 2) => true
--     is_simple_power(3, 2) => false
--     is_simple_power(3, 1) => false
--     is_simple_power(5, 3) => false
--     """
--     if (n == 1): 
--         return (x == 1) 
--     power = 1
--     while (power < x): 
--         power = power * n 
--     return (power == x) 
-- 


-- Haskell Implementation:

-- Your task is to write a function that returns true if a number x is a simple
-- power of n and false in other cases.
-- x is a simple power of n if n**int=x
-- For example:
-- is_simple_power 1 4 => true
-- is_simple_power 2 2 => true
-- is_simple_power 8 2 => true
-- is_simple_power 3 2 => false
-- is_simple_power 3 1 => false
-- is_simple_power 5 3 => false

is_simple_power :: Int -> Int -> Bool
is_simple_power x n =  or [n ^ i == x |  i <- [0..x]]



-- Test suite
check :: Bool -> IO ()
check True  = return ()
check False = error "Test failed"

main :: IO ()
main = do
    check (is_simple_power 1 4 == True)
    check (is_simple_power 2 2 == True)
    check (is_simple_power 8 2 == True)
    check (is_simple_power 3 2 == False)
    check (is_simple_power 3 1 == False)
    check (is_simple_power 5 3 == False)
    check (is_simple_power 16 2 == True)
    check (is_simple_power 143214 16 == False)
    check (is_simple_power 4 2 == True)
    check (is_simple_power 9 3 == True)
    check (is_simple_power 16 4 == True)
    check (is_simple_power 24 2 == False)
    check (is_simple_power 128 4 == False)
    check (is_simple_power 12 6 == False)
    check (is_simple_power 1 1 == True)
    check (is_simple_power 1 12 == True)
